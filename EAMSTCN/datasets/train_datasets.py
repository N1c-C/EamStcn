"""Data set for the pre-training phase of EAM-STM model. Following STM, and STCN and several other papers fake video
sequences are created from still image sets. An image is replicated three times. Affine transforms are then applied  to
simulate object movements. The Albumentations package is used instead of torchvision transforms. """

import PIL.Image as Image
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
from os import path
from EAMSTCN.datasets.imagenet_values import image_mean as image_mean, image_normalisation
import random


def reseed(seed):
    random.seed(seed)
    torch.manual_seed(seed)


class TrainImgDataSet(Dataset):
    def __init__(self, root_dir, height=240, width=320, pf=1):
        """
        The root dir should contain two folders one called images and the called masks.
        :param root_dir: str - the path to the directory containing the images and masks
        :param norm_trans: Bool: Apply Albumentations transforms: mean and std dev for ImageNet data
        :param trans: Bool: apply Albumentations transforms: size (DAVIS 480, 854) plus random flip\flop
        :param ff_trans: Bool: apply separate Albumentations transforms to first frame - Rotation, resize to IMAGE_WIDTH
        and IMAGE_HEIGHT, convert to tensor
        :param affine_trans: Bool: apply Albumentations affine transforms - random : rotation, sheering, zooming,
        translation, & cropping. Convert to tensor.
        """

        self.H = height
        self.W = width
        self.pf = pf
        self.root_dir = root_dir
        self.images = [img for img in os.listdir(root_dir + 'images') if not img.startswith('.')]  # get the image names
        self.normalise = A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0)

        self.transforms = A.Compose([
            A.Resize(height=int(1.5 * self.H), width=int(1.5 * self.W), interpolation=0),
            A.HorizontalFlip(p=0.3),
            A.VerticalFlip(p=0.3)])

        self.first_frame = A.Compose([
            A.SafeRotate(limit=(-15.0 * self.pf, 15.0 * self.pf), border_mode=0, value=image_mean, mask_value=0, p=0.4),
            A.Resize(height=height, width=width, interpolation=0, p=1.0),
            A.ColorJitter(brightness=0.3 * self.pf, contrast=0.3 * self.pf, saturation=0.3 * self.pf, hue=0, p=0.7),
            self.normalise,
            ToTensorV2()])

        self.affine = A.Compose([
            A.ColorJitter(brightness=0.3 * self.pf, contrast=0.3 * self.pf, saturation=0.3 * self.pf,
                          hue=0, p=0.7 * self.pf),
            A.Affine(scale=(1-(0.2 * self.pf), 1+(0.2 * self.pf)), translate_px=(0, 80 * self.pf),
                     shear=(-15.0 * pf, 15.0 * self.pf), cval=image_mean, cval_mask=0, mode=0,
                     fit_output=False, p=0.8, interpolation=0),
            A.SafeRotate(limit=(-15.0 * self.pf, 15.0 * self.pf), border_mode=0, value=image_mean, mask_value=0, p=0.6),
            A.RandomCrop(height, width, p=0.4),
            A.Resize(height=height, width=width, interpolation=0, p=1.0),
            self.normalise,
            ToTensorV2()])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        """
        Returns a pseudo sequence of 3 frames from a single image.
        :return: dict images {'seq': image seq, 'gt_seq: mask
        """
        images = {}
        im = np.array(Image.open(self.root_dir + 'images/' + self.images[index]).convert('RGB'))
        gt = np.array(Image.open(self.root_dir + 'masks/' + self.mask_name(self.images[index])).convert('L'))
        gt = (gt > 0.5).astype('uint8')  # convert mask to 1's & 0's

        # Apply random transforms to image and mask

        transformed = self.transforms(image=im, mask=gt)
        im = transformed['image']
        gt = transformed['mask']

        # Create a synthetic 3 frame video sequence if affine transforms supplied
        im1 = im2 = im
        gt1 = gt2 = gt

        if self.first_frame is not None:
            ff = self.first_frame(image=im, mask=gt)
        else:
            ff = self.affine(image=im, mask=gt)

        affine1 = self.affine(image=im1, mask=gt1)
        affine2 = self.affine(image=im2, mask=gt2)
        # affine = self.affine(image=im, mask=gt)
        images['seq'] = torch.cat([ff['image'].unsqueeze(0),
                                   affine1['image'].unsqueeze(0),
                                   affine2['image'].unsqueeze(0)], dim=0).float()

        images['gt1_seq'] = torch.cat([ff['mask'].reshape(1, 1, self.H, self.W),
                                       affine1['mask'].reshape(1, 1, self.H, self.W),
                                       affine2['mask'].reshape(1, 1, self.H, self.W)], dim=0).float()
        images['num_objs'] = 1
        images['gt2_seq'] = torch.zeros_like(images['gt1_seq']).float()
        images['selector'] = torch.FloatTensor([1, 0])
        images['cls_gt'] = torch.zeros(3, self.H, self.W, dtype=torch.long)
        images['cls_gt'][images['gt1_seq'][:, 0] > 0.5] = 1
        images['cls_gt'][images['gt2_seq'][:, 0] > 0.5] = 2

        return images

    def mask_name(self, image_name):
        """Since the different datasets have different naming conventions get mask is provided
        as an overload function to deal with the requirements of different datasets.
         If the mask and image share the same name the parent class is sufficient"""
        return image_name


class PascalDataSet(TrainImgDataSet):

    def mask_name(self, image_name):
        return image_name.replace('_im', '_gt')


class DutsDataSet(TrainImgDataSet):

    def mask_name(self, image_name):
        return image_name.replace('.jpg', '.png')


class EcssdDataSet(TrainImgDataSet):

    def mask_name(self, image_name):
        return image_name.replace('.jpg', '.png')


class HrsodDataSet(TrainImgDataSet):

    def mask_name(self, image_name):
        return image_name.replace('.jpg', '.png')


class CarsDataSet(TrainImgDataSet):

    def __getitem__(self, index, pf=1):
        """
        Returns a pseudo sequence of 3 frames from a single image.
        :return: dict images {'seq': image seq, 'gt_seq: mask
        """
        images = {}
        im = np.array(Image.open(self.root_dir + 'images/' + self.images[index]).convert('RGB'))
        gt = np.array(Image.open(self.root_dir + 'masks/' + self.mask_name(self.images[index])).convert('L'))
        gt = (gt > 0.5).astype('uint8')  # convert mask to 1's & 0's

        affine = A.Compose([
            A.Affine(scale=(1-(0.10 * pf), 1+(0.10 * pf)), translate_px=(0, 12 * pf), shear=(-10.0 * pf, 10.00 * pf), cval=image_mean, cval_mask=0,
                     mode=0, fit_output=False, p=0.70 * pf),
            A.SafeRotate(limit=(-10.00 * pf, 10.00 * pf), border_mode=0, value=image_mean, mask_value=0, p=0.60 * pf),
            # A.RandomCrop(self.H, self.W, p=0.4),
            self.normalise,
            ToTensorV2()])

        x = A.Compose([A.Resize(height=self.H, width=self.W, p=1.0)])
        ten = A.Compose([self.normalise, ToTensorV2()])
        # affine = self.affine(image=im, mask=gt)

        t = x(image=im, mask=gt)
        im1 = im2 = im = t['image']
        gt1 = gt2 = gt = t['mask']

        t = ten(image=im, mask=gt)
        im = t['image'].unsqueeze(0)
        gt = t['mask'].reshape(1, 1, self.H, self.W)

        t = affine(image=im1, mask=gt1)
        im1 = t['image'].unsqueeze(0)
        gt1 = t['mask'].reshape(1, 1, self.H, self.W)

        t = affine(image=im2, mask=gt2)
        im2 = t['image'].unsqueeze(0)
        gt2 = t['mask'].reshape(1, 1, self.H, self.W)

        images['seq'] = torch.cat([im, im1, im2], dim=0).float()
        images['gt1_seq'] = torch.cat([gt, gt1, gt2], dim=0).float()
        images['num_objs'] = 1
        images['gt2_seq'] = torch.zeros_like(images['gt1_seq']).float()
        images['selector'] = torch.FloatTensor([1, 0])
        images['cls_gt'] = torch.zeros(3, self.H, self.W, dtype=torch.long)
        images['cls_gt'][images['gt1_seq'][:, 0].squeeze() > 0.5] = 1
        images['cls_gt'][images['gt2_seq'][:, 0].squeeze() > 0.5] = 2
        return images

    def mask_name(self, image_name):
        return image_name.replace('.jpg', '_mask.gif')


# from PIL import Image
# import numpy as np


class VOSTrainDataset(Dataset):
    """ Modified from STCN and STM to read the DAVIS imset text file for train and val data sets
    It Picks three frames and two objects applies augmentation and controls the max distance between frames
    """

    def __init__(self, root, imset='2017', max_jump=5, height=384, width=384, pf=1, subset=None, youtube=False):
        """

        :param im_rot:
        :param gt_rot:
        :param root:
        :param imset:
        :param max_jump:
        :param height:
        :param width:
        :param pf:
        :param subset:
        """
        self.root = root
        # self.mask_dir = path.join(root, 'Annotations', '480p')
        # self.image_dir = path.join(root, 'JPEGImages', '480p')
        _imset_dir = path.join(root, 'ImageSets', imset)
        _imset_f = path.join(_imset_dir, 'train.txt')
        self.im_root = path.join(root, 'JPEGImages', '480p') if not youtube else path.join(root, 'JPEGImages')
        self.gt_root = path.join(root, 'Annotations', '480p') if not youtube else path.join(root, 'Annotations')
        self.height = height
        self.width = width
        self.max_jump = max_jump
        self.videos = []
        self.frames = {}
        self.pf = pf
        self.vid_list =[]
        if not youtube:
            with open(path.join(_imset_f), "r") as lines:
                for line in lines:
                    _video = line.rstrip('\n')
                    self.vid_list.append(_video)
            # vid_list = sorted([d for d in os.listdir(self.im_root) if not d.startswith('.')])
            # Pre-filtering
        else:
            self.vid_list = sorted([seq for seq in os.listdir(self.im_root) if not seq.startswith('.')])
        for vid in self.vid_list:
            if subset is not None:
                if vid not in subset:
                    continue
            frames = sorted([f for f in os.listdir(os.path.join(self.im_root, vid)) if not f.startswith('.')])
            if len(frames) < 3:
                continue
            self.frames[vid] = frames
            self.videos.append(vid)

        # These set of transform is the same for im/gt pairs, but different among the 3 sampled frames
        self.pair_im_lone_transform = transforms.Compose([
            transforms.ColorJitter(0.01 * self.pf, 0.01 * self.pf, 0.01 * self.pf, 0),
        ])

        self.pair_im_dual_transform = transforms.Compose([
            transforms.RandomAffine(degrees=15 * self.pf, shear=15 * self.pf, interpolation=InterpolationMode.BICUBIC, fill=image_mean),
        ])

        self.pair_gt_dual_transform = transforms.Compose([
            transforms.RandomAffine(degrees=15 * self.pf, shear=15 * self.pf, interpolation=InterpolationMode.NEAREST, fill=0),
        ])

        # These transform are the same for all pairs in the sampled sequence
        self.all_im_lone_transform = transforms.Compose([
            transforms.ColorJitter(0.1 * self.pf, 0.03 * self.pf, 0.03 * self.pf, 0),
            transforms.RandomGrayscale(0.05 * self.pf),
        ])

        self.all_im_dual_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop((height, width), scale=(0.36 * self.pf, 1.00 * self.pf), interpolation=InterpolationMode.BICUBIC)
        ])

        self.all_gt_dual_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop((height, width), scale=(0.36 * self.pf, 1.00 * self.pf), interpolation=InterpolationMode.NEAREST)
        ])

        # Final transform without randomness
        self.final_im_transform = transforms.Compose([
            transforms.ToTensor(),
            image_normalisation,
        ])

    def __getitem__(self, idx):
        video = self.videos[idx]
        info = {'name': video}

        vid_im_path = path.join(self.im_root, video)
        vid_gt_path = path.join(self.gt_root, video)
        frames = self.frames[video]

        trials = 0
        while trials < 5:
            info['frames'] = []  # Appended with actual frames

            # Don't want to bias towards beginning/end
            this_max_jump = min(len(frames), self.max_jump)
            start_idx = np.random.randint(len(frames) - this_max_jump + 1)
            f1_idx = start_idx + np.random.randint(this_max_jump + 1) + 1
            f1_idx = min(f1_idx, len(frames) - this_max_jump, len(frames) - 1)

            f2_idx = f1_idx + np.random.randint(this_max_jump + 1) + 1
            f2_idx = min(f2_idx, len(frames) - this_max_jump // 2, len(frames) - 1)

            frames_idx = [start_idx, f1_idx, f2_idx]
            if np.random.rand() < 0.5:
                # Reverse time
                frames_idx = frames_idx[::-1]

            sequence_seed = np.random.randint(2147483647)
            images = []
            masks = []
            target_object = None
            for f_idx in frames_idx:
                jpg_name = frames[f_idx][:-4] + '.jpg'
                png_name = frames[f_idx][:-4] + '.png'
                info['frames'].append(jpg_name)

                reseed(sequence_seed)
                this_im = Image.open(path.join(vid_im_path, jpg_name)).convert('RGB')
                this_im = self.all_im_dual_transform(this_im)
                this_im = self.all_im_lone_transform(this_im)
                reseed(sequence_seed)
                this_gt = Image.open(path.join(vid_gt_path, png_name)).convert('P')
                this_gt = self.all_gt_dual_transform(this_gt)

                pairwise_seed = np.random.randint(2147483647)
                reseed(pairwise_seed)
                this_im = self.pair_im_dual_transform(this_im)
                this_im = self.pair_im_lone_transform(this_im)
                reseed(pairwise_seed)
                this_gt = self.pair_gt_dual_transform(this_gt)

                this_im = self.final_im_transform(this_im)
                this_gt = np.array(this_gt)

                images.append(this_im)
                masks.append(this_gt)

            images = torch.stack(images, 0)

            labels = np.unique(masks[0])
            # Remove background
            labels = labels[labels != 0]

            if len(labels) == 0:
                target_object = -1  # all black if no objects
                has_second_object = False
                trials += 1
            else:
                target_object = np.random.choice(labels)
                has_second_object = (len(labels) > 1)
                if has_second_object:
                    labels = labels[labels != target_object]
                    second_object = np.random.choice(labels)
                break

        masks = np.stack(masks, 0)
        tar_masks = (masks == target_object).astype(np.float32)[:, np.newaxis, :, :]
        if has_second_object:
            sec_masks = (masks == second_object).astype(np.float32)[:, np.newaxis, :, :]
            selector = torch.FloatTensor([1, 1])
        else:
            sec_masks = np.zeros_like(tar_masks)
            selector = torch.FloatTensor([1, 0])

        cls_gt = np.zeros((3, self.height, self.width), dtype=np.int)
        cls_gt[tar_masks[:, 0] > 0.5] = 1
        cls_gt[sec_masks[:, 0] > 0.5] = 2

        data = {
            'seq': images,
            'gt1_seq': tar_masks,
            'cls_gt': cls_gt,
            'gt2_seq': sec_masks,
            'selector': selector,
            'info': info,
        }

        return data

    def __len__(self):
        return len(self.videos)


if __name__ == '__main__':

    inv_normalise = A.Compose([
        A.Normalize(
            mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
            std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
            max_pixel_value=255
        )])

    DUTS_PATH = '/Users/Papa/train/DUTS/'
    PASCAL_PATH = '/Users/Papa/train/pascal/'
    ECSSD_PATH = '/Users/Papa/train/ECSSD/'
    HRSOD_PATH = '/Users/Papa/train/HRSOD/'
    CARS_PATH = '/Users/Papa/cars/train/'
    VAL_PATH = '/Users/Papa/cars/val/'
    GT_ROOT = '/Users/Papa/DAVIS_2_obj/trainval/Annotations/480p/'
    IM_ROOT = '/Users/Papa/DAVIS_2_obj/trainval/JPEGImages/480p/'
    ROOT = '/Users/Papa/DAVIS_2_obj/trainval/'
    IMAGE_HEIGHT = 240
    IMAGE_WIDTH = 360

    prog_train_list = [()]
    pascal_data = PascalDataSet(PASCAL_PATH, height=IMAGE_HEIGHT, width=IMAGE_WIDTH, pf=1)
    ecssd_data = EcssdDataSet(ECSSD_PATH)
    davis_data = VOSTrainDataset(ROOT, max_jump=10, height=240, width=320, pf=1.2)

    cars_data = CarsDataSet(CARS_PATH, height=IMAGE_HEIGHT, width=IMAGE_WIDTH, pf=0.1)
    cars_val_data = CarsDataSet(VAL_PATH,  height=IMAGE_HEIGHT, width=IMAGE_WIDTH)

    pascal = DataLoader(pascal_data, batch_size=3, num_workers=2, pin_memory=False, shuffle=True)
    ecssd = DataLoader(ecssd_data, batch_size=3, num_workers=2, pin_memory=False, shuffle=True)
    davis = DataLoader(davis_data, batch_size=3, num_workers=2, pin_memory=False, shuffle=True)

    cars = DataLoader(cars_data, batch_size=3, num_workers=2, pin_memory=False, shuffle=True)

    data = iter(davis)
    for i in range(1):
        im = data.next()

        # print(im['gt1_seq'].shape)
        seq = im['seq'].cpu().permute(0, 1, 3, 4, 2).numpy()
        gts = im['gt1_seq'].cpu().permute(0, 1, 3, 4, 2).numpy()
        fig = plt.figure(f" Sample images", figsize=(20, 16))
        plt.axis("off")
        for x, i in enumerate(range(0, 18, 6)):
            # create a subplot
            # seq shape = B, T, C, H, W

            for frame in range(0, 3):
                plt.subplot(6, 3, i + 1 + frame)
                denorm = inv_normalise(image=seq[x][frame] * 255)
                plt.imshow(denorm['image'])
            for frame in range(0, 3):
                plt.subplot(6, 3, i + 1 + 3 + frame)
                plt.imshow(gts[x][frame].astype("uint8"))

            # image = image.transpose((1, 2, 0))

            # grab the label id and get the label from the classes list
            # idx = batch[1][i]
            # label = classes[idx]
            # show the image along with the label
            # plt.imshow(image)
            # plt.title(label)

        # show the plot
        plt.tight_layout()
        plt.show()
