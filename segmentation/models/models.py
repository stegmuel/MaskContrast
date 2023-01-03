#
# Authors: Wouter Van Gansbeke & Simon Vandenhende
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange
import math

"""
    SimpleSegmentationModel
    A simple encoder-decoder based segmentation model. 
"""


class SimpleSegmentationModel(nn.Module):
    def __init__(self, p, backbone, decoder, upsample_size=448):
        super(SimpleSegmentationModel, self).__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.upsample_size = upsample_size
        self.arch = p['arch']
        self.n_last_blocks = p['n_last_blocks']

    def forward(self, x):
        input_shape = x.shape[-2:]
        if "vit" in self.arch:
            intermediate_output = self.backbone.get_intermediate_layers(x, self.n_last_blocks)
            x = torch.cat([x[:, 1:] for x in intermediate_output], dim=-1)
        elif "swin" in self.arch:
            intermediate_output = self.backbone.forward_return_n_last_stages(x, self.n_last_blocks)

            x = []
            for i_o in intermediate_output:
                i_o = F.interpolate(i_o, size=28, mode='bilinear', align_corners=False)
                x.append(i_o)
            x = torch.cat(x, dim=1)
        else:
            x = self.backbone(x)
        x = self.decoder(x)
        if self.upsample_size is not None:
            # x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
            x = F.interpolate(x, size=self.upsample_size, mode='bilinear', align_corners=False)
        return x


class ContrastiveSegmentationModel(nn.Module):
    def __init__(self, backbone, decoder, head, upsample, use_classification_head=False, freeze_batchnorm='none'):
        super(ContrastiveSegmentationModel, self).__init__()
        self.backbone = backbone
        self.upsample = upsample
        self.use_classification_head = use_classification_head
        
        if head == 'linear': 
            # Head is linear.
            # We can just use regular decoder since final conv is 1 x 1.
            self.head = decoder[-1]
            decoder[-1] = nn.Identity()
            self.decoder = decoder

        else:
            raise NotImplementedError('Head {} is currently not supported'.format(head))


        if self.use_classification_head: # Add classification head for saliency prediction
            self.classification_head = nn.Conv2d(self.head.in_channels, 1, 1, bias=False)

    def forward(self, x):
        # Standard model
        input_shape = x.shape[-2:]
        x = self.backbone(x)
        embedding = self.decoder(x)

        # Head
        x = self.head(embedding)
        if self.use_classification_head:
            sal = self.classification_head(embedding)

        # Upsample to input resolution
        if self.upsample: 
            x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
            if self.use_classification_head:
                sal = F.interpolate(sal, size=input_shape, mode='bilinear', align_corners=False)

        # Return outputs
        if self.use_classification_head:
            return x, sal.squeeze()
        else:
            return x
