import torch
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
        self.model.memory.top_k = top_k
        self.save_every = save_every
        self.save_pr_fr = save_pr_fr
        self.model.memory.num_objects = num_objs
        self.model.memory.max_size = memory_size
        self.preds = torch.zeros(self.num_objs + 1, self.num_of_frs, 1, self.pad_h, self.pad_w)
        self.preds[0] = 1e-7

        # self.kh = self.pad_h // 16
        # self.kw = self.pad_w // 16

        # purge any existing memory
        self.model.memory.clear()

    def evaluate(self, gt_mask, first_fr_idx, last_fr_idx):
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

        # prev_mask = self.preds[:, first_fr_idx].to(self.device)
        # last_diff_val = 0

        end = last_fr_idx - 1  # Flag to halt loop before last frame

        for fr in range(first_fr_idx + 1, last_fr_idx):  # start loop on next frame
            q_features, key_q = self.model.eval_encode_key(self.images[:, fr].to(self.device))
            # key_q = key_q.unsqueeze(2)
            pred_mask = self.model.get_mask_for_frame(q_features, key_q)

            pred_mask = F.softmax(aggregate(pred_mask, dim=0, keep_bg=True), dim=0)
            self.preds[:, fr] = pred_mask

            if fr != end:
                # im_diff_val = self.model.memory.calc_img_var(self.images[:, fr-1], prev_mask[1:], self.images[:, fr],
                #                                              pred_mask[1:], last_diff_val)
                # prev_mask = pred_mask.to(self.device)

                is_mem_fr = ((fr % self.save_every) == 0)
                # is_mem_fr = (save > 0.8)
                # if self.save_pr_fr or is_mem_fr or abs(im_diff_val - last_diff_val) >= 800:
                if self.save_pr_fr or is_mem_fr:  # or abs(im_diff_val) > 100
                    mem_v = self.model.eval_encode_value(self.images[:, fr].to(self.device),
                                                         pred_mask[1:].to(self.device))
                    mem_key = key_q.unsqueeze(2)
                    self.model.memory.add_to_memory(mem_key, mem_v)

                # last_diff_val = im_diff_val
        print("frames saved:", self.model.memory.mem_size())
        return last_fr_idx
