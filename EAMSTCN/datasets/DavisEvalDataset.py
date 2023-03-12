"""
Modified from https://github.com/seoungwugoh/STM/blob/master/dataset.py
"""

import os
from os import path
import numpy as np
from PIL import Image
import torch
from torch.utils.data.dataset import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from EAMSTCN.utils import one_hot_mask


class DAVISEvalDataset(Dataset):
    def __init__(self, root, imset='2017/val.txt', size=480, target_name=None):
        self.root = root
        self.mask_dir = path.join(root, 'Annotations', '480p')
        self.mask480_dir = path.join(root, 'Annotations', '480p')
        self.image_dir = path.join(root, 'JPEGImages', '480p')
        _imset_dir = path.join(root, 'ImageSets')
        _imset_f = path.join(_imset_dir, imset)

        self.videos = []
        self.num_frames = {}
        self.num_objects = {}
        self.shape = {}
        self.size_480p = {}
        with open(path.join(_imset_f), "r") as lines:
            for line in lines:
                _video = line.rstrip('\n')
                if target_name is not None and target_name != _video:
                    continue
                self.videos.append(_video)
                self.num_frames[_video] = len([f for f in os.listdir(path.join(self.image_dir, _video)) if not f.startswith('.')])
                _mask = np.array(Image.open(path.join(self.mask_dir, _video, '00000.png')).convert("P"))
                self.num_objects[_video] = np.max(_mask)
                self.shape[_video] = np.shape(_mask)
                _mask480 = np.array(
                    Image.open(path.join(self.mask480_dir, _video, '00000.png')).convert("P"))
                self.size_480p[_video] = np.shape(_mask480)

        self.im_transform = A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, index):
        video = self.videos[index]
        info = {'name': video, 'frames': [], 'num_frames': self.num_frames[video], 'size': self.size_480p[video]}
        images = []
        masks = []
        for f in range(self.num_frames[video]):
            img_file = path.join(self.image_dir, video, '{:05d}.jpg'.format(f))
            # img = np.array(Image.open(img_file).resize((self.width, self.height)).convert('RGB'))
            img = np.array(Image.open(img_file).convert('RGB'))
            im = self.im_transform(image=img)
            images.append(im['image'])
            info['frames'].append('{:05d}.jpg'.format(f))

            mask_file = path.join(self.mask_dir, video, '{:05d}.png'.format(f))
            if path.exists(mask_file):
                masks.append(np.array(Image.open(mask_file).convert('P'),
                                      dtype=np.uint8))
                # masks.append(np.array(Image.open(mask_file).resize((self.width, self.height)).convert('P'),
                #                       dtype=np.uint8))
            else:
                # fill with empty masks when no ground truth
                masks.append(np.zeros_like(masks[0]))

        images = torch.stack(images, 0)
        masks = np.stack(masks, 0)
        gts = masks
        masks, info['labels'] = one_hot_mask(torch.from_numpy(masks), start_label=1)
        masks = masks.unsqueeze(2)
        info['gt_obj'] = {0: np.array([1])}
        data = {
            'seq': images,
            'gt_seq': masks,
            'info': info,
            'gts': gts
        }
        return data
