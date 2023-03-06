"""
Different decoder classes  with increasing numbers of channels for exploration of gains versus the penalty
"""


import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import FeaturePyramidNetwork as Fpn
from EAMSTCN.modules import ResBlock, UpsampleBlock


class FpnDecoder(nn.Module):
    """ Takes a set of conv feature maps from consecutive output stages and feeds into a feature pyramid
    network based on 'Feature Pyramid Network for Object Detection' paper . The final stage and interpolation are
    the same as the STM code"""

    def __init__(self, in_channels, fpn_channels, fpn_stage=2):
        super().__init__()
        print(fpn_channels)
        fpn_channels = 0
        self.fpn = Fpn(in_channels, fpn_channels)
        # self.decon = nn.ConvTranspose2d(in_channels=fpn_channels, out_channels=fpn_channels, kernel_size=2, stride=2)
        self.head_out = nn.Conv2d(in_channels=fpn_channels, out_channels=1, kernel_size=3, padding='same')
        self.stage = fpn_stage

    def forward(self, key_features):
        h, w = key_features['stage_1'].shape[-2:]

        x = self.fpn(key_features)
        x = self.head_out(F.silu(x[f'stage_{self.stage}']))
        x = F.interpolate(x, scale_factor=2 ** self.stage, mode='bilinear', align_corners=False)

        # Perform un-padding on the fly. The  features will be padded by a max of two in any either direction
        mh, mw = x.shape[-2:]
        if mh != h * 2 or mw != w * 2:
            x = unpad(x, mh, mw, h * 2, w * 2)
        return x


class StcnDecoder(nn.Module):
    """ Parent Decoder Class. STCN"""

    def __init__(self, channels):
        super().__init__()
        self.compress = ResBlock(channels[-1], channels[-2])
        self.up_16_8 = UpsampleBlock(channels[-2], channels[-2], channels[-3])  # 1/16 -> 1/8
        self.up_8_4 = UpsampleBlock(channels[-3], channels[-3], channels[-3])  # 1/8 -> 1/4

        self.pred = nn.Conv2d(channels[-3], 1, kernel_size=(3, 3), padding=(1, 1), stride=1)

    def forward(self, features):
        x = self.compress(features['stage_4'])
        x = self.up_16_8(features['stage_3'], x)
        x = self.up_8_4(features['stage_2'], x)
        x = self.pred(F.relu(x))

        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)
        return x


class Decoder1(StcnDecoder):

    def __init__(self, channels):
        print(channels)
        super().__init__()
        self.compress = ResBlock(channels[-1], 512)
        self.up_16_8 = UpsampleBlock(channels[-2], 512, 256)  # 1/16 -> 1/8
        self.up_8_4 = UpsampleBlock(channels[-3], 256, 256)  # 1/8 -> 1/4

        self.pred = nn.Conv2d(256, 1, kernel_size=(3, 3), padding=(1, 1), stride=1)


class Decoder2(StcnDecoder):

    def __init__(self, channels):
        print(channels)
        super().__init__()
        self.compress = ResBlock(channels[-1], 512)
        self.up_16_8 = UpsampleBlock(channels[-2], 512, 512)  # 1/16 -> 1/8
        self.up_8_4 = UpsampleBlock(channels[-3], 512, 512)  # 1/8 -> 1/4

        self.pred = nn.Conv2d(512, 1, kernel_size=(3, 3), padding=(1, 1), stride=1)


def unpad(msk, mh, mw, h, w):
    """ Lose any padding when not using stage1 fpn output.
    Not always required, it depends on the input size """
    if mh > h and not mw > w:
        return msk[:, :, 1:-1, :]
    elif not mh > h and mw > w:
        return msk[:, :, :, 1:-1]
    else:
        return msk[:, :, 1:-1, 1:-1]
