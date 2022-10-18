#
# Authors: Wouter Van Gansbeke & Simon Vandenhende
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

import os
import random

import cv2
import data.dataloaders.custom_transforms as custom_tr
import numpy as np
import torch
import torchvision.transforms as transforms
from einops.layers.torch import Rearrange
from models import vision_transformer as vits
from utils.collate import collate_custom

from utils import dino_utils


def load_pretrained_weights(p, model):
    # Load weights from pre-training
    print('Loading pre-trained weights from {}'.format(p['pretraining']))
    state_dict = torch.load(p['pretraining'], map_location='cpu')['model']
    new_state = {}

    for k, v in state_dict.items():
        if k.startswith('module.model_q.'):
            new_state[k.rsplit('module.model_q.')[1]] = v
        else:
            pass

    msg = model.load_state_dict(new_state, strict=False)
    print('Loading state dict from checkpoint')
    print('Warning: This piece of code was only tested for linear classification')
    print('Warning: Assertions should probably depend on model type (Segm/ContrastiveSegm)')
    assert (set(msg[0]) == set(['decoder.4.weight', 'decoder.4.bias']))
    assert (set(msg[1]) == set(['head.weight', 'head.bias', 'classification_head.weight']))

    # Init final conv layer
    if 'deeplab' in p['head']:
        model.decoder[4].weight.data.normal_(mean=0.0, std=0.01)
        model.decoder[4].bias.data.zero_()


class ResNet50Bottom(torch.nn.Module):
    # https://forums.fast.ai/t/pytorch-best-way-to-get-at-intermediate-layers-in-vgg-and-resnet/5707/2
    def __init__(self, original_model):
        super(ResNet50Bottom, self).__init__()
        # Remove avgpool and fc layers
        self.features = torch.nn.Sequential(*list(original_model.children())[:-2])

    def forward(self, x):
        x = self.features(x)
        return x


def get_model(p):
    # Get backbone
    if "resnet" in p['backbone']:
        if p['resnet_dilate'] == 1:
            replace_stride_with_dilation = [False, False, False]
        elif p['resnet_dilate'] == 2:
            replace_stride_with_dilation = [False, False, True]
        elif p['resnet_dilate'] == 4:
            replace_stride_with_dilation = [False, True, True]

    if p['backbone'] == 'resnet18':
        import torchvision.models.resnet as resnet
        backbone = resnet.__dict__['resnet18'](pretrained=False,
                                               replace_stride_with_dilation=replace_stride_with_dilation)
        backbone_channels = 512

    elif p['backbone'] == 'resnet50':
        import torchvision.models.resnet as resnet
        backbone = resnet.__dict__['resnet50'](pretrained=False,
                                               replace_stride_with_dilation=replace_stride_with_dilation)
        backbone_channels = 2048

        # Load pretrained weights
        dino_utils.load_pretrained_weights_resnet(backbone, p['pretraining'])
        backbone = ResNet50Bottom(backbone)
    elif 'vit' in p['backbone']:
        backbone = vits.__dict__[p['arch']](patch_size=p['patch_size'])
        backbone_channels = backbone.embed_dim
        # Load pre-trained weights
        dino_utils.load_pretrained_weights(backbone, p['pretraining'], p['checkpoint_key'], p['arch'], p['patch_size'])
    else:
        raise ValueError('Invalid backbone {}'.format(p['backbone']))

    if p['dilated'] and 'resnet' in p['backbone']:
        from models.resnet_dilated import ResnetDilated
        backbone = ResnetDilated(backbone)

    # Get head
    if p['head'] == 'deeplab':
        if not p['kmeans_eval']:
            nc = p['num_classes'] + int(p['has_bg'])
        else:
            nc = p['model_kwargs']['ndim']

        from models.deeplab import DeepLabHead
        head = DeepLabHead(backbone_channels, nc)

    elif p['head'] == 'dim_reduction':
        nc = p['num_classes'] + int(p['has_bg'])
        import torch.nn as nn
        head = nn.Conv2d(backbone_channels, nc, 1)
    elif p['head'] == 'linear':
        nc = p['num_classes'] + int(p['has_bg'])
        import torch.nn as nn
        if 'vit' in p['backbone']:
            head = nn.Sequential(
                Rearrange(pattern='b (h w) d -> b d h w', h=28, w=28),
                nn.Conv2d(backbone_channels, nc, 1)
            )
        else:
            head = nn.Sequential(
                nn.Conv2d(backbone_channels, nc, 1)
            )
    elif p['head'] == 'identity':
        import torch.nn as nn
        head = nn.Sequential(
            Rearrange(pattern='b (h w) d -> b d h w', h=28, w=28),
        )
    else:
        raise ValueError('Invalid head {}'.format(p['head']))

    # Compose model from backbone and head
    if p['kmeans_eval']:
        from models.models import ContrastiveSegmentationModel
        import torch.nn as nn
        model = ContrastiveSegmentationModel(backbone, head, p['model_kwargs']['head'],
                                             p['model_kwargs']['upsample'],
                                             p['model_kwargs']['use_classification_head'], p['freeze_layer'])
    else:
        from models.models import SimpleSegmentationModel
        model = SimpleSegmentationModel(backbone, head, p['embeddings_upsample'])

        # Load pretrained weights
        # load_pretrained_weights(p, model)
    return model


def get_train_dataset(p, transform=None):
    if p['train_db_name'] == 'VOCSegmentation':
        from data.dataloaders.pascal_voc import VOC12
        dataset = VOC12(root=p['data_path'], split=p['split'], transform=transform, masks_upsample=p['masks_upsample'])
    elif 'coco' in p['train_db_name']:
        from data.dataloaders.coco import COCOSegmentation
        dataset_name = p['train_db_name']
        assert len(dataset_name.split("_")) == 2
        mask_type = dataset_name.split("_")[-1]
        assert mask_type in ["thing", "stuff"]
        file_list = os.listdir(os.path.join(p['data_path'], "images", "train2017"))
        # file_list_val = os.listdir(os.path.join(p['data_path'], "coco", "images", "val2017"))
        # random.shuffle(file_list_val)

        # sample 10% of train images
        random.shuffle(file_list)
        file_list = file_list[:int(len(file_list) * 0.1)]
        print(f"sampled {len(file_list)} COCO images for training")

        dataset = COCOSegmentation(
            root=p['data_path'],
            file_names=file_list,
            mask_type=mask_type,
            image_set='train',
            transforms=transform,
            masks_upsample=p['masks_upsample']
        )
    else:
        raise ValueError('Invalid train dataset {}'.format(p['train_db_name']))
    return dataset


def get_val_dataset(p, transform=None):
    if p['val_db_name'] == 'VOCSegmentation':
        from data.dataloaders.pascal_voc import VOC12
        dataset = VOC12(root=p['data_path'], split='val', transform=transform, masks_upsample=p['masks_upsample'])
    elif 'coco' in p['train_db_name']:
        from data.dataloaders.coco import COCOSegmentation
        dataset_name = p['train_db_name']
        assert len(dataset_name.split("_")) == 2
        mask_type = dataset_name.split("_")[-1]
        assert mask_type in ["thing", "stuff"]
        file_list_val = os.listdir(os.path.join(p['data_path'], "images", "val2017"))
        random.shuffle(file_list_val)

        dataset = COCOSegmentation(
            root=p['data_path'],
            file_names=file_list_val,
            mask_type=mask_type,
            image_set='val',
            transforms=transform,
            masks_upsample=p['masks_upsample']
        )
    else:
        raise ValueError('Invalid validation dataset {}'.format(p['val_db_name']))
    return dataset


def get_train_dataloader(p, dataset):
    return torch.utils.data.DataLoader(dataset, num_workers=p['num_workers'],
                                       batch_size=p['batch_size'], pin_memory=True,
                                       collate_fn=collate_custom, drop_last=True, shuffle=True)


def get_val_dataloader(p, dataset):
    return torch.utils.data.DataLoader(dataset, num_workers=p['num_workers'],
                                       batch_size=p['batch_size'], pin_memory=True,
                                       collate_fn=collate_custom, drop_last=False, shuffle=False)


def get_train_transformations(augmentation_strategy='pascal'):
    return transforms.Compose([custom_tr.RandomHorizontalFlip(),
                               custom_tr.ScaleNRotate(rots=(-5, 5), scales=(.75, 1.25),
                                                      flagvals={'semseg': cv2.INTER_NEAREST, 'image': cv2.INTER_CUBIC}),
                               custom_tr.FixedResize(
                                   resolutions={'image': tuple((448, 448)), 'semseg': tuple((448, 448))},
                                   flagvals={'semseg': cv2.INTER_NEAREST, 'image': cv2.INTER_CUBIC}),
                               custom_tr.ToTensor(),
                               custom_tr.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


def get_val_transformations():
    return transforms.Compose([custom_tr.FixedResize(resolutions={'image': tuple((448, 448)),
                                                                  'semseg': tuple((448, 448))},
                                                     flagvals={'image': cv2.INTER_CUBIC, 'semseg': cv2.INTER_NEAREST}),
                               custom_tr.ToTensor(),
                               custom_tr.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


def get_optimizer(p, parameters):
    if p['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(parameters, lr=p['lr'], weight_decay=p['weight_decay'], momentum=p['momentum'],
                                    nesterov=p['nesterov'])

    elif p['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(parameters, lr=p['lr'])

    else:
        raise ValueError('Invalid optimizer {}'.format(p['optimizer']))

    return optimizer


def adjust_learning_rate(p, optimizer, epoch):
    lr = p['lr']
    if p['scheduler'] == 'step':
        steps = np.sum(epoch > np.array(p['lr_decay_epochs']))
        if steps > 0:
            lr = lr * (p['lr_decay_rate'] ** steps)

    elif p['scheduler'] == 'poly':
        lambd = pow(1 - (epoch / p['epochs']), 0.9)
        lr = lr * lambd

    else:
        raise ValueError('Invalid learning rate schedule {}'.format(p['scheduler']))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr
