#
# Authors: Wouter Van Gansbeke & Simon Vandenhende
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

import argparse
import cv2
import os
import numpy as np
import torch
import torch.nn as nn

from utils.config import update_config
from utils.common_config import get_val_dataset, get_val_transformations,\
                                get_val_dataloader,\
                                get_model
from utils.kmeans_utils import save_embeddings_to_disk, eval_kmeans
from termcolor import colored
import torchvision.transforms as transforms
from segmentation.utils.dino_utils import bool_flag
from termcolor import colored

# Parser
# parser = argparse.ArgumentParser(description='Fully-supervised segmentation')
# parser.add_argument('--config_env', default='configs/env.yml', help='Config file for the environment')
# parser.add_argument('--config_exp',
#                     default='configs/kmeans/kmeans_VOCSegmentation_supervised_saliency.yml',
#                     help='Config file for the experiment')
# parser.add_argument('--num_seeds', default=5, type=int, help='number of seeds during kmeans')
# args = parser.parse_args()

def get_args_parser():
    parser = argparse.ArgumentParser('OC', add_help=False)
    parser.add_argument("--root_dir",
                        default='/home/thomas/Documents/phd/samno_paper/MaskContrast/output/OC',
                        type=str)
    parser.add_argument("--train_db_name", default="coco_thing", type=str)
    parser.add_argument("--data_path", default='/media/thomas/Elements/cv_datasets/coco/', type=str)
    parser.add_argument("--split", default='trainaug', type=str)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--val_db_name", default="coco_thing")
    parser.add_argument("--num_workers", default=6, type=int)
    parser.add_argument("--backbone", default='vit', type=str)
    parser.add_argument("--dilated", default=True, type=bool_flag)
    parser.add_argument("--pretrained", default=False, type=bool_flag)
    parser.add_argument("--head", default='identity', type=str)
    parser.add_argument("--arch", default='vit_small', type=str)
    parser.add_argument("--patch_size", default=16, type=int)
    parser.add_argument("--checkpoint_key", default='teacher', type=str)
    parser.add_argument("--pretraining", default='/home/thomas/Downloads/temp/multi_head_008/checkpoint0300.pth',
                        type=str)
    parser.add_argument("--freeze_batchnorm", default='all', type=str)
    parser.add_argument('--crf-postprocess', action='store_true', help='Apply CRF post-processing during evaluation')
    parser.add_argument('--num_seeds', default=5, type=int, help='number of seeds during kmeans')
    parser.add_argument('--embeddings_upsample', default=28, help='Apply CRF post-processing during evaluation')
    parser.add_argument('--masks_upsample', default=100, help='Apply CRF post-processing during evaluation')
    parser.add_argument('--pca_dim', default=50, help='Number of dimensions after PCA.')
    parser.add_argument("--n_last_blocks", default=4, type=int)
    parser.add_argument("--resnet_dilate", default=2, type=int)
    parser.add_argument("--n_clusters", default=500, type=int)
    return parser


def main(args):
    p = vars(args)
    cv2.setNumThreads(1)

    # Retrieve config file
    p = update_config(p)
    # p = create_config(args.config_env, args.config_exp)
    print('Python script is {}'.format(os.path.abspath(__file__)))
    print(colored(p, 'red'))

    # Get model
    print(colored('Retrieve model', 'blue'))
    model = get_model(p)
    model = model.cuda()

    # CUDNN
    print(colored('Set CuDNN benchmark', 'blue')) 
    torch.backends.cudnn.benchmark = True

    # Dataset
    print(colored('Retrieve dataset', 'blue'))
    
    # Transforms 
    from data.dataloaders.pascal_voc import VOC12
    val_transforms = get_val_transformations()
    val_dataset = get_val_dataset(p, val_transforms)
    # val_dataset = VOC12(root=p['val_db_kwargs']['path'], split='val', transform=val_transforms)
    val_dataloader = get_val_dataloader(p, val_dataset)

    # true_val_dataset = VOC12(root=p['val_db_kwargs']['path'], split='val', transform=None)
    true_val_dataset = get_val_dataset(p, None)
    print(colored('Val samples %d' %(len(true_val_dataset)), 'yellow'))

    # Kmeans Clustering
    # n_clusters = p['n_clusters']
    n_clusters = p['num_classes'] + int(p['has_bg'])
    results_miou = []
    for i in range(args.num_seeds):
        save_embeddings_to_disk(p, val_dataloader, model, n_clusters=n_clusters, seed=1234 + i, pca_dim=args.pca_dim)
        eval_stats = eval_kmeans(p, true_val_dataset, n_clusters=n_clusters, verbose=True)
        results_miou.append(eval_stats['mIoU'])

        # Write results
        with open(os.path.join(p['output_dir'], f'{i}_mIoU_results.txt'), 'w') as file:
            file.write(f"mIoU: {eval_stats['mIoU'] * 100}")
    print(colored('Average mIoU is %2.1f' %(np.mean(results_miou)*100), 'green'))

    # Write only the mIoU
    with open(os.path.join(p['output_dir'], 'average_mIoU_results.txt'), 'w') as file:
        file.write(f"mIoU: {np.mean(results_miou) * 100}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser('OC', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
