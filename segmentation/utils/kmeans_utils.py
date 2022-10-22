#
# Authors: Wouter Van Gansbeke & Simon Vandenhende
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

import os
import cv2
import numpy as np
import torch
from PIL import Image
from utils.utils import SemsegMeter
from sklearn.cluster import MiniBatchKMeans
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA
from termcolor import colored
from joblib import Parallel, delayed
from sklearn import metrics
from einops import rearrange
from utils.leopart_utils import normalize_and_transform, cluster

N_JOBS = 16 # set to number of threads


def eval_kmeans(p, val_dataset, n_clusters=21, compute_metrics=False, verbose=True):
    n_classes = p['num_classes'] + int(p['has_bg'])

    # Iterate
    tp = [0] * n_classes
    fp = [0] * n_classes
    fn = [0] * n_classes
  
    # Load all pixel embeddings
    all_pixels = np.zeros((len(val_dataset) * 500 * 500), dtype=np.float32)
    all_gt = np.zeros((len(val_dataset) * 500 * 500), dtype=np.float32)
    offset_ = 0

    for i, sample in enumerate(val_dataset):
        if i % 300 == 0:
            print('Evaluating: {} of {} objects'.format(i, len(val_dataset)))

        # Load embedding
        filename = os.path.join(p['embedding_dir'], sample['meta']['image'] + '.npy')
        embedding = np.load(filename)

        # Check where ground-truth is valid. Append valid pixels to the array.
        gt = sample['semseg']
        valid = (gt != 255)
        n_valid = np.sum(valid)
        all_gt[offset_:offset_+n_valid] = gt[valid]

        # Possibly reshape embedding to match gt.
        if embedding.shape != gt.shape:
            embedding = cv2.resize(embedding, gt.shape[::-1], interpolation=cv2.INTER_NEAREST)

        # Put the reshaped ground truth in the array
        all_pixels[offset_:offset_+n_valid,] = embedding[valid]
        all_gt[offset_:offset_+n_valid,] = gt[valid]

        # Update offset_
        offset_ += n_valid

    # All pixels, all ground-truth
    all_pixels = all_pixels[:offset_,]
    all_gt = all_gt[:offset_,]

    # Do hungarian matching
    print(colored('Starting hungarian', 'green'))
    num_elems = offset_
    if n_clusters == n_classes:
        print('Using hungarian algorithm for matching')
        match = _hungarian_match(all_pixels, all_gt, preds_k=n_clusters, targets_k=n_classes, metric='iou')

    else:
        print('Using majority voting for matching')
        match = _majority_vote(all_pixels, all_gt, preds_k=n_clusters, targets_k=n_classes)

    # Remap predictions
    reordered_preds = np.zeros(num_elems, dtype=all_pixels.dtype)
    for pred_i, target_i in match:
        reordered_preds[all_pixels == int(pred_i)] = int(target_i)

    if compute_metrics:
        print('Computing acc, nmi, ari ...')
        acc = int((reordered_preds == all_gt).sum()) / float(num_elems)
        nmi = metrics.normalized_mutual_info_score(all_gt, reordered_preds)
        ari = metrics.adjusted_rand_score(all_gt, reordered_preds)
    else: 
        acc, nmi, ari = None, None, None

    # TP, FP, and FN evaluation
    print(colored('Starting miou', 'green'))
    for i_part in range(0, n_classes):
        tmp_all_gt = (all_gt == i_part)
        tmp_pred = (reordered_preds == i_part)
        tp[i_part] += np.sum(tmp_all_gt & tmp_pred)
        fp[i_part] += np.sum(~tmp_all_gt & tmp_pred)
        fn[i_part] += np.sum(tmp_all_gt & ~tmp_pred)

    jac = [0] * n_classes
    for i_part in range(0, n_classes):
        jac[i_part] = float(tp[i_part]) / max(float(tp[i_part] + fp[i_part] + fn[i_part]), 1e-8)

    # Write results
    eval_result = dict()
    eval_result['jaccards_all_categs'] = jac
    eval_result['mIoU'] = np.mean(jac)
    eval_result['acc'] = acc
    eval_result['nmi'] = nmi
    eval_result['ari'] = ari
        
    if verbose:
        print('Evaluation of semantic segmentation ')
        print('mIoU is %.2f' %(100*eval_result['mIoU']))
        class_names = val_dataset.get_class_names()
        for i_part in range(n_classes):
            print('IoU class %s is %.2f' %(class_names[i_part], 100*jac[i_part]))

    print(eval_result)

    return eval_result


@torch.no_grad()
def save_embeddings_to_disk(p, val_loader, model, n_clusters=21, seed=1234, pca_dim=32):
    print('Save embeddings to disk ...')
    model.eval()

    all_prototypes = []
    names = []
    for i, batch in enumerate(val_loader):
        output = model(batch['image'].cuda(non_blocking=True)).cpu()
        meta = batch['meta']

        output = rearrange(output, 'b d h w -> (b h w) d')
        all_prototypes.append(output)
        for name in meta['image']:
            names.append(name)

        if i % 10 == 0:
            print(f"Computing prototype {i}/{len(val_loader)}")

    # perform kmeans
    all_prototypes = torch.cat(all_prototypes)
    print('Kmeans clustering to {} clusters'.format(n_clusters))
    
    print(colored('Starting kmeans with scikit', 'green'))
    all_prototypes = normalize_and_transform(all_prototypes, pca_dim=pca_dim)
    prediction_kmeans = cluster(pca_dim, all_prototypes.numpy(), 28, n_clusters, seed=seed)

    # save predictions
    for i, fname, pred in zip(range(len(val_loader.sampler)), names, prediction_kmeans):
        # prediction = all_sals[i].copy()
        # prediction[prediction == 1] = pred + 1
        np.save(os.path.join(p['embedding_dir'], fname + '.npy'), pred)
        if i % 300 == 0:
            print('Saving results: {} of {} objects'.format(i, len(val_loader.dataset)))


def _hungarian_match(flat_preds, flat_targets, preds_k, targets_k, metric='acc'):
    assert (preds_k == targets_k)  # one to one
    num_k = preds_k

    # perform hungarian matching
    print('Using iou as metric')
    results = Parallel(n_jobs=N_JOBS, backend='multiprocessing')(delayed(get_iou)(flat_preds, flat_targets, c1, c2) for c2 in range(num_k) for c1 in range(num_k))
    results = np.array(results)
    results = results.reshape((num_k, num_k)).T
    match = linear_sum_assignment(flat_targets.shape[0] - results)
    match = np.array(list(zip(*match)))
    res = []
    for out_c, gt_c in match:
        res.append((out_c, gt_c))
    return res


def _majority_vote(flat_preds, flat_targets, preds_k, targets_k):
    iou_mat = Parallel(n_jobs=N_JOBS, backend='multiprocessing')(delayed(get_iou)(flat_preds, flat_targets, c1, c2) for c2 in range(targets_k) for c1 in range(preds_k))
    iou_mat = np.array(iou_mat)
    results = iou_mat.reshape((targets_k, preds_k)).T
    results = np.argmax(results, axis=1)
    match = np.array(list(zip(range(preds_k), results)))
    return match


def get_iou(flat_preds, flat_targets, c1, c2):
    tp = 0
    fn = 0
    fp = 0
    tmp_all_gt = (flat_preds == c1)
    tmp_pred = (flat_targets == c2)
    tp += np.sum(tmp_all_gt & tmp_pred)
    fp += np.sum(~tmp_all_gt & tmp_pred)
    fn += np.sum(tmp_all_gt & ~tmp_pred)
    jac = float(tp) / max(float(tp + fp + fn), 1e-8)
    return jac
