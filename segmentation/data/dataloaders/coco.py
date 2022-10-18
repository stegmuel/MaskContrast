"""
Source: https://github.com/MkuuWaUjinga/leopart/
"""
import json
import os
import cv2
from typing import List, Optional, Callable, Tuple, Any

import numpy as np
import torch
from PIL import Image
from torchvision.datasets import VisionDataset


class COCOSegmentation(VisionDataset):

    def __init__(
            self,
            root: str,
            file_names: List[str],
            mask_type: str,
            masks_upsample: int,
            image_set: str = "train",
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            transforms: Optional[Callable] = None,
    ):
        super(COCOSegmentation, self).__init__(root, transforms, transform, target_transform)
        self.image_set = image_set
        self.file_names = file_names
        self.mask_type = mask_type
        self.masks_upsample = masks_upsample
        assert self.image_set in ["train", "val"]
        assert mask_type in ["stuff", "thing"]

        # Set mask folder depending on mask_type
        if mask_type == "thing":
            # seg_folder = "annotations/panoptic_annotations/semantic_segmentation_{}2017/"
            # seg_folder = "annotations/panoptic_annotations_trainval2017/annotations/panoptic_{}2017"
            seg_folder = "/home/thomas/Downloads/{}2017"
            json_file = "annotations/panoptic_annotations_trainval2017/annotations/panoptic_train2017.json"
        elif mask_type == "stuff":
            # seg_folder = "annotations/stuff_annotations_trainval2017/annotations/"
            seg_folder = "annotations/stuff_annotations_trainval2017/annotations/stuff_{}2017_pixelmaps/"
            json_file = "annotations/stuff_annotations_trainval2017/annotations/stuff_val2017.json"
        else:
            raise ValueError(f"No support for image set {self.image_set}")
        seg_folder = seg_folder.format(image_set)
        # json_file = json_file.format(image_set)

        # Load categories to category to id map for merging to coarse categories
        with open(os.path.join(root, json_file)) as f:
            an_json = json.load(f)
            all_cat = an_json['categories']
            if mask_type == "thing":
                all_thing_cat_sup = set(cat_dict["supercategory"] for cat_dict in all_cat if cat_dict["isthing"] == 1)
                super_cat_to_id = {super_cat: i for i, super_cat in enumerate(sorted(all_thing_cat_sup))}
                self.cat_id_map = {}
                for cat_dict in all_cat:
                    if cat_dict["isthing"] == 1:
                        self.cat_id_map[cat_dict["id"]] = super_cat_to_id[cat_dict["supercategory"]]
                    elif cat_dict["isthing"] == 0:
                        self.cat_id_map[cat_dict["id"]] = 255
            else:
                super_cats = set([cat_dict['supercategory'] for cat_dict in all_cat])
                super_cats.remove("other")  # remove others from prediction targets as this is not semantic
                super_cat_to_id = {super_cat: i for i, super_cat in enumerate(sorted(super_cats))}
                super_cat_to_id["other"] = 255  # ignore_index for CE
                self.cat_id_map = {cat_dict['id']: super_cat_to_id[cat_dict['supercategory']] for cat_dict in all_cat}

        # Get images and masks fnames
        seg_dir = os.path.join(root, seg_folder)
        image_dir = os.path.join(root, "images", f"{image_set}2017")
        if not os.path.isdir(seg_dir) or not os.path.isdir(image_dir):
            print(seg_dir)
            print(image_dir)
            raise RuntimeError('Dataset not found or corrupted.')
        self.images = [os.path.join(image_dir, x) for x in self.file_names]
        self.masks = [os.path.join(seg_dir, x.replace("jpg", "png")) for x in self.file_names]
        self.id_to_class = {v: k for k, v in super_cat_to_id.items()}

    def get_class_names(self):
        return self.id_to_class

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        # Load the data
        img = np.array(Image.open(self.images[index]).convert('RGB'))
        mask = np.array(Image.open(self.masks[index]))

        # Prepare the mask and meta data
        sample = {}
        if mask.shape[0] != self.masks_upsample or mask.shape[1] != self.masks_upsample:
            mask = cv2.resize(mask, [self.masks_upsample, self.masks_upsample], interpolation=cv2.INTER_NEAREST)

        sample['meta'] = {'im_size': (img.shape[0], img.shape[1]),
                          'image_file': self.images[index],
                          'image': os.path.basename(self.masks[index]).split('.')[0]}
        sample['image'] = img

        if self.mask_type == "stuff":
            # move stuff labels from {0} U [92, 183] to [0,15] and [255] with 255 == {0, 183}
            # (183 is 'other' and 0 is things)
            # mask *= 255
            assert np.max(mask) <= 183
            mask[mask == 0] = 183  # [92, 183]
            assert np.min(mask) >= 92
            for cat_id in np.unique(mask):
                mask[mask == cat_id] = self.cat_id_map[cat_id]

            assert np.max(mask) <= 255
            assert np.min(mask) >= 0
            # mask /= 255
            # return img, mask
            sample['semseg'] = mask
        elif self.mask_type == "thing":
            # mask *= 255
            # assert np.max(mask) <= 200
            # print(np.unique(mask))
            mask[mask == 0] = 200  # map unlabelled to stuff
            mask[mask > 200] = 200
            merged_mask = mask.copy()
            for cat_id in np.unique(mask):
                try:
                    merged_mask[mask == cat_id] = self.cat_id_map[int(cat_id)]  # [0, 11] + {255}
                except KeyError:
                    # Catch the classes with missing labels
                    merged_mask[mask == cat_id] = 255

            assert np.max(merged_mask) <= 255
            assert np.min(merged_mask) >= 0
            # merged_mask /= 255
            # return img, merged_mask
            sample['semseg'] = merged_mask
        # TODO: add meta

        if self.transforms is not None:
            sample = self.transforms(sample)
        return sample