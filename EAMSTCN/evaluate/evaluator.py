"""
Wrapper Class to evaluate EamStcn on DAVIS-17 and YouTube VOS2018 Datasets. THe data gen is required to provide
a dict called in info["gt_obj"] where the key is the frame number of a GT Mask. For the DAVIS set it is always the
first frame but for the YouTube set GT_Masks pop up at different points. Therefore, the value for the key is a tensor
listing the objects that are in the GT annotation (DAVIS must have at least '1' in the tensor but doesn't need
all the objects).

The data should be in the form of:    number of objects (k), frames in seq(T), C, H, W
"""

import torch
from EAMSTCN.utils import *
import torch.nn.functional as F


class EvalEamStcn:
    def __init__(self, model, images, num_objs, num_of_frs, info, top_k=20, save_every=5,
                 memory_size=None, device='cpu', adaptive=False):
        """
        :param model:  Trained EamStcn torch model
        :param images: Batch of images from datagen
        :param num_objs: Int: The number of objects in the sequence to segment
        :param num_of_frs: Int: Total number of frames to segment
        :param info: Dictionary passed from the data gen that holds seq names, gt info, frame names etc
        :param top_k: Int: The value of k to be used in top-k affinity calculation
        :param save_every: Int: The save rate of the network
        :param memory_size: Int: If employing the LRU sets the maximum number of features that can be stored in mem
        :param device: Torch device to store tensor, 'cuda', 'mps', or 'cpu'
        :param adaptive: Bool - Choose adaptive saving or a fixed save rate. When adaptive save set - the save rate acts
        as the ASR, ideally set to either 25 or 10
        """
        self.model = model
        # Store the original dimensions
        self.t = images.shape[1]
        self.h, self.w = images.shape[-2:]
        self.info = info
        images, self.pad = pad_divide_by(images, 16)
        self.images = images
        self.device = device
        self.num_objs = num_objs
        self.num_of_frs = num_of_frs
        self.pad_h, self.pad_w = images.shape[-2:]
        self.model.memory.top_k = top_k
        self.save_every = save_every
        self.model.memory.num_objects = num_objs
        self.model.memory.max_size = memory_size
        self.preds = torch.zeros(self.num_objs + 1, self.num_of_frs, 1, self.pad_h, self.pad_w)
        self.preds[0] = 1e-7
        self.adaptive = adaptive
        self.msk_aoc = []  # Stores the amount of pixel change between predicted masks if adaptive saving
        self.im_aoc = []  # Stores the amount of pixel change between predicted frames if adaptive saving

        # purge any existing memory
        self.model.memory.clear()

    def evaluate(self, gt_mask, first_fr_idx, last_fr_idx):
        """"""
        gt_mask, _ = pad_divide_by(gt_mask.to(self.device), 16)

        end = last_fr_idx - 1  # Flag to halt loop before last frame

        # Inialise vars for determining the rate of change for the mask and image
        prev_mask = self.preds[:, first_fr_idx].to(self.device)
        prev_im_diff = 0
        prev_msk_diff = 0

        max_save_rate = 4  # set to n+1 to force n frames before the next one can be saved
        save_cntdown = 1  # Counter - Tracks the number of frames since the last saved frame to maintain max_save_rate
        trigger_frame = False  # Set by the trigger function, When False the Save Rate determines if a frame is stored
        saved = False  # Flag to indicate a GT frame is already saved

        for fr in range(first_fr_idx, last_fr_idx):
            save_cntdown -= 1 if save_cntdown > 0 else 0
            # Look for the annotation frames
            if fr in self.info['gt_obj']:
                saved = True
                # if 1 in (self.info['gt_obj'][fr].tolist()[0]):
                if fr == 0:
                    # If it's the first one then we add it to predictions and the memory
                    self.add_to_preds_and_memory(self.images[:, fr], gt_mask[:, fr], fr)

                else:
                    # If it is later we predict first object(s) and then merge with the new gt
                    # The data generator should provide a long torch tensor with the object numbers which we use as
                    # indices to select the correct mask tensor

                    q_features, q_key = self.model.eval_encode_key(self.images[:, fr].to(self.device))
                    pred_mask = self.model.get_mask_for_frame(q_features, q_key)

                    # -1 from the object(s) for indexing purposes
                    i = torch.sub(self.info['gt_obj'][fr], 1)
                    pred_mask[i.long()] = gt_mask[:, fr][i.long()]
                    self.add_to_preds_and_memory(self.images[:, fr], pred_mask, fr)
                    pred_mask = self.preds[:, fr].to(self.device)  # we need the aggregate mask incase adaptive saving

            # # Otherwise predict the mask
            else:
                q_features, q_key = self.model.eval_encode_key(self.images[:, fr].to(self.device))

                # For YouTube there might not be a first frame, so we make a blank mask
                if self.model.memory.mem_size() == 0:
                    pred_mask = torch.zeros(self.num_objs, 1, self.pad_h, self.pad_w)
                else:
                    pred_mask = self.model.get_mask_for_frame(q_features, q_key)

                pred_mask = F.softmax(aggregate(pred_mask, dim=0, keep_bg=True), dim=0)
                self.preds[:, fr] = pred_mask

            if fr != end:
                # self.images shape B, T, C, H, W

                # If adaptive saving then find the difference between images
                if self.adaptive and fr != 0:
                    im_diff, msk_diff = self.model.memory.calc_img_var(self.images[0, fr - 1],
                                                                       prev_mask[1:],
                                                                       self.images[0, fr],
                                                                       pred_mask[1:].to(self.device))

                    # Call the save trigger
                    trigger_frame = self.mem_trigger(im_diff, msk_diff, prev_im_diff, prev_msk_diff)

                # If the first frame has no gt object we still need to start the memory for later matching
                if fr == 0 and not saved:
                    self.to_mem(self.images[:, fr], pred_mask[1:], q_key)
                else:
                    # Determine if the save every x frames condition is true
                    is_mem_fr = ((fr % self.save_every) == 0) if fr != 0 else False  #  We don't want to over

                    # Save to memory when either condition is met and not already saved
                    if save_cntdown == 0:
                        if (is_mem_fr or trigger_frame) and not saved:
                            self.to_mem(self.images[:, fr], pred_mask[1:], q_key)
                            save_cntdown = max_save_rate

            # Save the mask for the next frames calculation when adaptive saving
            prev_mask = self.preds[:, fr].to(self.device)
            saved = False  # Reset the flag

        print("frames saved:", self.model.memory.mem_size())
        return last_fr_idx

    def to_mem(self, im, mask, q_key):
        mem_v = self.model.eval_encode_value(im.to(self.device), mask.to(self.device))
        mem_key = q_key.unsqueeze(2)
        self.model.memory.add_to_memory(mem_key, mem_v)

    def add_to_preds_and_memory(self, im, mask, fr):
        self.preds[:, fr] = F.softmax(aggregate(mask, dim=0, keep_bg=True), dim=0)
        q_features, q_key = self.model.eval_encode_key(im.to(self.device))
        self.to_mem(im, self.preds[1:, fr], q_key)

    def mem_trigger(self, im_diff, msk_diff, prev_im_diff, prev_msk_diff):
        """
        Compare the absolute amount of image difference between the current and previous frames
        :param im_diff: float: a single value representing the pixel difference between two frames
        :param msk_diff: float: a single value representing the pixel difference between two masks
        :param prev_im_diff: float: a single value representing the pixel difference between the previous two frames
        :param prev_msk_diff: float: a single value representing the pixel difference between the previous two frames
        :return: Bool: Save the frame if True
        """
        self.im_aoc.append(abs(abs(im_diff) - abs(prev_im_diff)))
        self.msk_aoc.append(abs(abs(msk_diff) - abs(prev_msk_diff)))
        # Find the amount of change (roc) between frames
        msk_roc = im_roc = 0
        if len(self.im_aoc) > 1:
            im_roc = abs(self.im_aoc[-1] - self.im_aoc[-2])
            msk_roc = abs(self.msk_aoc[-1] - self.msk_aoc[-2])

        # Simple set of conditions to force a frame to be saved
        if msk_roc > .3:
            return True

        # important one
        if msk_roc < 0.1 and im_roc > 0.5:
            return True

        if self.num_objs >= 3 and msk_roc > .1:
            return True

        if msk_roc > 0.25 and im_roc > .25:
            return True

        if msk_roc > 0.15 and im_roc > .6:
            return True

        return False

if __name__ == '__main__':
    x = torch.zeros(3,3,3)
    i=torch.tensor([2,3])
    y=torch.rand(3,3,3)

    print(x,y,i)

    i = (torch.sub(i, 1))
    print(i.type())
    x[i] = y[i]
    print(x)