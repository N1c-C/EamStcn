""" Dataset for the Youtube Vos Test dataset -  Modified from Modified from
https://github.com/seoungwugoh/STM/blob/master/yv_text.py"""

import os
from os import path

import matplotlib.pyplot as plt
import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
import numpy as np

from EAMSTCN.datasets.imagenet_values import image_mean as image_mean, image_normalisation

from EAMSTCN.utils import one_hot_mask


def all_to_onehot(masks, labels):
    if len(masks.shape) == 3:
        Ms = np.zeros((len(labels), masks.shape[0], masks.shape[1], masks.shape[2]), dtype=np.uint8)
    else:
        Ms = np.zeros((len(labels), masks.shape[0], masks.shape[1]), dtype=np.uint8)

    for k, l in enumerate(labels):
        Ms[k] = (masks == l).astype(np.uint8)

    return Ms


class YouTubeTestDataset(Dataset):
    def __init__(self, data_root, res=480):

        self.image_dir = data_root + '/JPEGImages'
        self.mask_dir = data_root + '/Annotations/'

        self.videos = []
        self.shape = {}
        self.frames = {}

        vid_list = sorted(f for f in os.listdir(self.image_dir) if not f.startswith('.'))

        # Pre-reading
        for vid in vid_list:
            frames = sorted(os.listdir(os.path.join(self.image_dir, vid)))
            self.frames[vid] = frames

            self.videos.append(vid)
            first_mask = os.listdir(path.join(self.mask_dir, vid))[0]
            _mask = np.array(Image.open(path.join(self.mask_dir, vid, first_mask)).convert("P"))
            self.shape[vid] = np.shape(_mask)

        if res != -1:
            self.im_transform = transforms.Compose([
                transforms.ToTensor(),
                image_normalisation,
                transforms.Resize(res, interpolation=InterpolationMode.BICUBIC),
            ])

            self.mask_transform = transforms.Compose([
                transforms.Resize(res, interpolation=InterpolationMode.NEAREST),
            ])
        else:
            self.im_transform = transforms.Compose([
                transforms.ToTensor(),
                image_normalisation,
            ])

            self.mask_transform = transforms.Compose([
            ])

    def __getitem__(self, idx):
        video = self.videos[idx]
        info = {}
        info['name'] = video
        info['frames'] = self.frames[video]
        info['size'] = self.shape[video]  # Real sizes
        info['gt_obj'] = {}  # Frames with labelled objects

        vid_im_path = path.join(self.image_dir, video)
        vid_gt_path = path.join(self.mask_dir, video)

        frames = self.frames[video]

        images = []
        masks = []
        for i, f in enumerate(frames):
            img = Image.open(path.join(vid_im_path, f)).convert('RGB')
            images.append(self.im_transform(img))

            mask_file = path.join(vid_gt_path, f.replace('.jpg', '.png'))
            if path.exists(mask_file):
                masks.append(np.array(Image.open(mask_file).convert('P'), dtype=np.uint8))
                # plt.imshow(np.array(Image.open(mask_file).convert('P')))
                # plt.show()
                this_labels = np.unique(masks[-1])
                this_labels = this_labels[this_labels != 0]
                info['gt_obj'][i] = this_labels
            else:
                # Mask not exists -> nothing in it
                masks.append(np.zeros(self.shape[video]))

        images = torch.stack(images, 0)
        masks = np.stack(masks, 0)

        # Construct the forward and backward mapping table for labels
        # this is because YouTubeVOS's labels are sometimes not continuous
        # while we want continuous ones (for one-hot)
        # so we need to maintain a backward mapping table
        labels = np.unique(masks).astype(np.uint8)
        labels = labels[labels != 0]
        info['label_convert'] = {}
        info['label_backward'] = {}
        idx = 1
        for lb in labels:
            info['label_convert'][lb] = idx
            info['label_backward'][idx] = lb
            idx += 1
        masks = torch.from_numpy(all_to_onehot(masks, labels)).float()
        gts = masks
        # Resize to 480p
        masks = self.mask_transform(masks)
        masks = masks.unsqueeze(2)

        info['labels'] = labels
        if max(labels) > len(labels):
            for i, k in enumerate(info['gt_obj'].keys()):
                info['gt_obj'][k] = i+1

        data = {
            'seq': images,
            'gt_seq': masks,
            'info': info,
            'gts': gts
        }

        return data

    def __len__(self):
        return len(self.videos)
