import torch

from EAMSTCN.metrics.ContoursAndMoments import get_moments, match_shape
from EAMSTCN.utils import *
import torch.nn.functional as F


class EvalEamStm:
    def __init__(self, model, images, num_objs, num_of_frs, height, width, top_k=20, save_every=5,
                 memory_size=None, save_pr_fr=False, device='cpu'):
        """"""
        self.model = model
        # Store the original dimensions
        self.t = images.shape[1]
        self.h, self.w = images.shape[-2:]

        images, self.pad = pad_divide_by(images, 16)
        self.images = images
        self.device = device
        self.num_objs = num_objs
        self.num_of_frs = num_of_frs
        self.pad_h, self.pad_w = images.shape[-2:]
        self.save_pr_fr = save_pr_fr
        self.model.memory.num_objects = num_objs
        self.model.memory.max_size = memory_size
        self.preds = torch.zeros(self.num_objs + 1, self.num_of_frs, 1, self.pad_h, self.pad_w)
        self.preds[0] = 1e-7

        self.model.memory.top_k = top_k
        self.avg_im_roc = []
        self.save_every = save_every
        self.save_pr_fr = save_pr_fr
        self.msk_aoc = []  # Stores the amount of pixel change between predicted masks
        self.im_aoc = []  # Stores the amount of pixel change between predicted frames
        # purge existing memory
        self.model.memory.clear()

    def evaluate(self, gt_mask, first_fr_idx, last_fr_idx):  # , fr_to_save
        """"""
        gt_mask, _ = pad_divide_by(gt_mask.to(self.device), 16)

        self.preds[:, first_fr_idx] = F.softmax(aggregate(gt_mask, dim=0, keep_bg=True), dim=0)

        # get the Query feature, key and value for the first frame
        q_features, q_key = self.model.eval_encode_key(self.images[:, first_fr_idx].to(self.device))
        q_key = q_key.unsqueeze(2)
        key_v = self.model.eval_encode_value(self.images[:, first_fr_idx].to(self.device),
                                             self.preds[1:, first_fr_idx].to(self.device))  # Lose the bground prob
        # add the keys to the memory bank
        self.model.memory.add_to_memory(q_key, key_v)

        # Prep for determining the rate of change for the mask and image
        prev_mask = self.preds[:, first_fr_idx].to(self.device)
        prev_im_diff = 0
        prev_msk_diff = 0

        end = last_fr_idx - 1  # Flag to halt loop before last frame
        save_cntdown = 1
        # start loop on next frame
        for fr in range(first_fr_idx + 1, last_fr_idx):
            q_features, key_q = self.model.eval_encode_key(self.images[:, fr].to(self.device))
            # key_q = key_q.unsqueeze(2)
            pred_mask = self.model.get_mask_for_frame(q_features, key_q)
            pred_mask = F.softmax(aggregate(pred_mask, dim=0, keep_bg=True), dim=0)
            self.preds[:, fr] = pred_mask
            save_cntdown -= 1 if save_cntdown > 0 else 0
            if fr != end:
                # self.images shape B, T, C, H, W
                im_diff, msk_diff = self.model.memory.calc_img_var(self.images[0, fr - 1], prev_mask[1:],
                                                                   self.images[0, fr],
                                                                   pred_mask[1:], prev_im_diff, prev_msk_diff)
                prev_mask = pred_mask.to(self.device)
                prev_key = key_q
                # use moments
                # self.save_every = save_frame2(match_shape(prev_mask, pred_mask[1:]))

                # Use Difference triggers
                save_frame = self.mem_trigger1(im_diff, msk_diff, prev_im_diff, prev_msk_diff)
                # self.save_every, save_frame = self.save_frame(im_diff, msk_diff, prev_im_diff, prev_msk_diff)
                # self.save_every = 25
                # print(self.save_every, save_frame)

                is_mem_fr = ((fr % self.save_every) == 0) if self.save_every != False else False

                # if save_frame:
                #     # if self.save_pr_fr or is_mem_fr:
                #     # match_shape(prev_mask, pred_mask[1:])
                #     # print('this frame saved')
                #     mem_v = self.model.eval_encode_value(self.images[:, fr-1].to(self.device),
                #                                          prev_mask[1:].to(self.device))
                #     # mem_v = self.model.eval_encode_value(self.images[:, fr].to(self.device),
                #     #                                      mask[:, fr].to(self.device))
                #     mem_key = prev_key.unsqueeze(2)
                #     self.model.memory.add_to_memory(mem_key, mem_v)

                # if self.save_pr_fr or is_mem_fr or save_frame:
                print(save_frame, fr)
                if save_cntdown == 0:
                    if self.save_pr_fr or is_mem_fr or save_frame:  # or is_mem_fr
                        # match_shape(prev_mask, pred_mask[1:])
                        # print('this frame saved')
                        mem_v = self.model.eval_encode_value(self.images[:, fr].to(self.device),
                                                             pred_mask[1:].to(self.device))
                        # mem_v = self.model.eval_encode_value(self.images[:, fr].to(self.device),
                        #                                      mask[:, fr].to(self.device))
                        mem_key = key_q.unsqueeze(2)
                        self.model.memory.add_to_memory(mem_key, mem_v)
                        save_cntdown = 4

        print("frames saved:", self.model.memory.mem_size())
        return last_fr_idx

    def mem_trigger2(self, im_diff, msk_diff, prev_im_diff, prev_msk_diff):
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
        # Find the amount of change (roc) between frames        msk_roc = im_roc = 0
        if len(self.im_aoc) > 1:
            im_roc = abs(self.im_aoc[-1] - self.im_aoc[-2])
            msk_roc = abs(self.msk_aoc[-1] - self.msk_aoc[-2])
            self.avg_im_roc.append(msk_roc.item())
            print(msk_roc.item(), im_roc.item(), self.average_roc())
            if msk_roc > 0.3525 and im_roc > 0.3525:
                # print('yes you got here')
                return True

    def mem_trigger1(self, im_diff, msk_diff, prev_im_diff, prev_msk_diff):
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

    def average_roc(self):
        """Finds the average rate of change
        currently not used"""
        return np.sum(self.avg_im_roc) / len(self.avg_im_roc)



