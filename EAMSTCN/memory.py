""" All functions and classes for the memory function of EAMSTCN"""

import math
import torch.nn.functional as F
import numpy as np
import torch
import PIL.Image as Image
import time
import torch.nn as nn


class MemoryBank:
    def __init__(self, n_obj=1, top_k=20, max_size=None, Train=False):
        """ Class to store features from previous frames to aid mask generation in future frames for a
        single video sequence. Modified from  https://github.com/hkchengrex/STCN
        :param n_obj: int the number of objects in the video
        :param top_k: int the k best matching results to be returned by the affinity function
        :param max_size: int the maximum number of frames in the memory before memory management starts.
        :var self.key_memory: is the memory for the key values
        :var self.value_memory: stores the mask values for feature fusion
        :var self.temp_key: Holds the previous frames key if including it in the look-up calculation
        :var self.temp_value: Holds the previous value if including it in the look-up calculation
        :var self.num_objects: The number of objects to segment in a video sequence
        :var self.max.size: The maximum number of frames that may be stored
        :var self.T: The current number of frames stored in the memory
        :var self.Ck, self.Cv: Hold the dimension size (number of channels) of the projected keys from the encoders
        """
        self.top_k = top_k
        self.Ck = None
        self.Cv = None
        self.h = None
        self.w = None
        self.key_memory = None
        self.value_memory = None
        self.temp_key = None
        self.temp_value = None
        self.num_objects = n_obj
        self.max_size = max_size
        self.T = 0
        self.lru = -1 * torch.ones(self.max_size) if self.max_size is not None else None

    def add_to_memory(self, key, value, is_temp=False):
        """ New feature frames have the shape (1, C, h, W) where h and w are 1/16 the size of
        the image dimensions.
        :param key:
        :param value:
        :param is_temp: Flag to determine if frame is used as memory
        """

        if self.key_memory is None:
            # Add the first frame
            _, self.Ck, _, self.h, self.w = key.shape
            _, self.Cv, _, _, _ = value.shape
            key = key.flatten(start_dim=2)
            value = value.flatten(start_dim=2)
            self.key_memory = key
            self.value_memory = value
            self.T += 1

        else:
            if is_temp:
                self.temp_key = key
                self.temp_value = value
            else:
                if self.lru is not None and self.T == self.max_size:
                    self.purge_lru()
                key = key.flatten(start_dim=2)
                value = value.flatten(start_dim=2)
                self.key_memory = torch.cat([self.key_memory, key], 2)
                self.value_memory = torch.cat([self.value_memory, value], 2)
                self.T += 1
        # print('\nKey Memory size is: ', self.key_memory.shape)
        # print('Val Memory size is: ', self.value_memory.shape, '\n')

    def affinity(self, key_mem_kM, q_key_kQ):
        """As per STCN paper equation 5 (last term discarded) - calculates the -ve squared euclidean distance between
        the memory key and query key as the similarity measure. The equation terms are included in the variable names
        for clarity though not pep-8 compatible.
        Expands -ve Euclidean distance - (a -b)^2 into  2ab - a^2 - b^2 : normalised by sqrt(1/Ck).
        The -b^2 term is ignored (see paper)
        This function could be replaced with dot product or cosine similarity functions but STCN authors found fewer
        memory entries contribute to the final mask with these methods resulting in lower accuracy
        Note: the dim values (sum and squeeze) are different to the paper's pseudocode to include the batch dimension"""
        # print('affinity gets flattened query key shape of : ', q_key_kQ.shape)
        # print('affinity gets  key mem shape of: ', key_mem_kM.shape, '\n')
        B, Ck, NE = key_mem_kM.shape  # NE: number of elements T*H*W: keys are flattened and concatenated into mem

        a_sq = key_mem_kM.pow(2).sum(1).unsqueeze(2)

        ab = key_mem_kM.transpose(1, 2) @ q_key_kQ

        affinity = (2 * ab - a_sq) / math.sqrt(Ck)  # B, NE, HW

        affinity = self._topk_soft_max(affinity, k=self.top_k)  # B, NE, HW

        return affinity

    # noinspection PyMethodMayBeStatic
    def _topk_soft_max(self, affinity, k):
        """
        Modified from STCN. Fn replaces the manual smax calculation with torch's inbuilt softmax function since the
        original code produced NaN values. Top k returns the k-largest elements of the affinity matrix along with their
        indices in the given dimension. The calculated softmax scores are then scattered back into the source with zeros
        replacing the other values.
        Uses the top k indices to time stamp the appropriate memory frame for the LRU cache
        :param affinity:
        :param k:
        :return:
        """
        top_k_values, indices = torch.topk(affinity, k=k, dim=1)

        if self.max_size is not None:
            self.update_lru(indices.unique())
        y = torch.softmax(top_k_values, dim=1)  # Tensor size  B * THW * HW
        affinity.zero_().scatter_(1, indices, y)

        return affinity

    # noinspection PyMethodMayBeStatic
    def _readout(self, affinity, val_mem_vM):
        """Performs equation 2 from STCN paper:
         vQ = vM.W  where W is the affinity matrix
         Uses batch matrix-matrix product
        tensors must be 3D of the shape  (b x n x m) (b x m x p) -> output shape (b x m x p)
        :Returns the weighted sum of memory features to be passed to merged with key encoder's 1/16
        feature set."""
        # print('readout aff: ', affinity.shape)
        # print('readout val mem: ', val_mem_vM.shape)
        return torch.bmm(val_mem_vM, affinity)

    def match_memory(self, q_key_kQ):
        """

        :param q_key_kQ:
        :return:
        """

        # print('match memory gets q-key shape of: ', q_key_kQ.shape, '\n')
        k = self.num_objects
        _, _, h, w = q_key_kQ.shape

        q_key_kQ = q_key_kQ.flatten(start_dim=2)

        if self.temp_key is not None:
            key_mem_kM = torch.cat([self.key_memory, self.temp_key], 2)
            val_mem_vM = torch.cat([self.value_memory, self.temp_value], 2)
        else:
            key_mem_kM = self.key_memory
            val_mem_vM = self.value_memory

        affinity = self.affinity(key_mem_kM, q_key_kQ)
        # print('affinity tensor', affinity.shape)
        # print('val_mem_vM ', val_mem_vM.shape, '\n')

        # single affinity tensor but expand it for the number of objects
        readout_mem = self._readout(affinity.expand(k, -1, -1), val_mem_vM)
        # print('readout_mem after Bmm affinity and val mem', readout_mem.shape)
        # print('readout_mem_final ', readout_mem.view(k, self.Cv, h, w).shape)
        return readout_mem.view(k, self.Cv, h, w)

    def update_lru(self, cols):
        """
        Time stamps features when they are used. The div effectively returns the frames positions
        in the flattened chain of keys
        "trunc" - rounds the results of the division towards zero. Equivalent to C-style integer division
        :param cols: is a tensor of the unique indices from the affinity calculation

        """
        hw = self.h * self.w
        print('frames used: ', torch.div(cols, hw, rounding_mode='trunc').unique())
        self.lru[(torch.div(cols, hw, rounding_mode='trunc').unique())] = time.time()  # equiv of old // division

    @staticmethod
    def calc_img_var(prev_frame, prev_mask, curr_frame, curr_mask, last_val=0, last_msk=0):
        """ Image variation calculation inspired by swiftnet: https://arxiv.org/abs/2102.04604 but a much simpler
        idea. Determines the amount of difference between frames  as a percentage of all the pixels. The value is used
        by adaptive evaluator mem_trigger_function
        :param prev_frame: The previous frame as a torch tensor
        :param prev_mask: The predicted mask for the previous frame - torch tensor
        :param curr_frame: The current input image
        :param curr_mask: The current predicted mask
        :return:
        """
        pixels = prev_frame.shape[1] * prev_frame.shape[2]

        img_diff = torch.sum((torch.sum(torch.abs(curr_frame - prev_frame), dim=0) > 4) / pixels) * 100
        msk_diff = torch.sum((torch.sum(torch.abs(curr_mask - prev_mask), dim=0) > 0.99) / pixels) * 100

        return img_diff, msk_diff




    @staticmethod
    def cal_img_norm(prev_frame, prev_mask, curr_frame, curr_mask):
        """Image variation calculation """
        img_diff = torch.linalg.norm(torch.sum((curr_frame - prev_frame), dim=1)) / 255
        # msk_diff = torch.linalg.norm((curr_mask) - (prev_mask)) / 255
        return img_diff

    def mem_size(self):
        return self.T

    def purge_lru(self):
        """

        :return:
        """
        _, t = self.lru.min(dim=0)
        self.purge(t.item())

    def purge(self, t):
        """
        Removes the given frame (t) from the feature memory
        :param t: The frame number to be removed
        :return: No return value
        """
        hw = (self.h * self.w)  # The height and width of the 1/16 features to give number of elements per key

        if t == self.T:  # the passed frame is the last one
            self.key_memory = self.key_memory[:, :, :-hw]
            self.value_memory = self.value_memory[:, :, :-hw]
        else:
            self.key_memory = torch.cat([
                self.key_memory[:, :, 0:t * hw],
                self.key_memory[:, :, t * hw + hw:]], 2)
            self.value_memory = torch.cat([
                self.value_memory[:, :, 0:t * hw],
                self.value_memory[:, :, t * hw + hw:]], 2)

    def clear(self):
        """Reset the memory"""
        self.value_memory = None
        self.key_memory = None
        self.Ck = None
        self.Cv = None
        self.h = None
        self.w = None
        self.T = 0
        self.lru = -1 * torch.ones(self.max_size) if self.max_size is not None else None


class TrainMemoryBank(MemoryBank):
    """Simplified memory function for training since there is no requirement to store frames
    Provides the functions to calculate the affinity and perform a readout of values on a
    passed tensor of values. Unlike evaluation memory bank all values are used for matching and not the top-k
    """

    def affinity(self, key_mem_kM, q_key_kQ):
        """ Overload function for training. As per STCN paper equation 5 (last term discarded).
         calculates -ve squared euclidean distance between the pseudo key memory and query key.
         Unlike the evaluation function the whole affinity matrix is used  - not the top k.
         Affinity is regulated
        :param key_mem_kM: Tensor of the form B, Ck, T, H, W
        :param q_key_kQ: Tensor of the form B, Ck, T, H, W
        :return: -ve euclidean distance similarity tensor """
        # print(key_mem_kM.shape, q_key_kQ.shape)
        B, Ck, T, H, W = key_mem_kM.shape  # NE: number of elements T*H*W  since keys are flattened and concatenated into mem

        key_mem_kM = key_mem_kM.flatten(start_dim=2)
        q_key_kQ = q_key_kQ.flatten(start_dim=2)

        a_sq = key_mem_kM.pow(2).sum(1).unsqueeze(2)

        ab = key_mem_kM.transpose(1, 2) @ q_key_kQ  # B THW  HW

        affinity = (2 * ab - a_sq) / math.sqrt(Ck)  # B, NE, HW
        maxes = torch.max(affinity, dim=1, keepdim=True)[0]  # ignore the indices tensor

        return torch.softmax((affinity - maxes), dim=1)



    def match_memory(self, affinity, val_memory_Vm, key_f16_features):
        """ Over load function, because for training we are performing memory matching on multiple sequences
        (batches) for just two frames. The function uses the affinity matrix with a temporary memory to extract the
        memorised value.
        and the precomputed key_f16_features to return the final f16 features for the decoder.
        :param affinity:
        :param val_memory_Vm:
        :param key_f16_features:
        :return: """
        # print('val mem shape match mem ', val_memory_Vm.shape, affinity.shape, key_f16_features.shape)
        B, Cv, T, H, W = val_memory_Vm.shape
        Vm = val_memory_Vm.view(B, Cv, T * H * W)
        memory_out = torch.bmm(Vm, affinity)  # Weighted-sum B, CV, HW
        memory_out = memory_out.view(B, Cv, H, W)
        mem_value_out = torch.cat([memory_out, key_f16_features], dim=1)
        return mem_value_out






