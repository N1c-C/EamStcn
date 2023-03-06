"""This is a modified factory script from https://github.com/abhuse/pytorch-efficientnet
It enables a PyTorch EfficientNetV2 model to be instantiated either for classification or segmentation.py
When creating a segmentation model, the final convolution stage and the original head layers are removed.
The output from the remaining convolution stages are fed into  a feature pyramid network FPN followed by a
final convolution layer for single channel mask predictions.

Use BCEWithLogitsLoss() loss function and seg=True for a segmentation model

Both options can be preloaded with the Imagenet weights for effective transfer learning"""

from collections import OrderedDict
from math import ceil

import torch
import torch.nn as nn
from torch.utils import model_zoo

from EAMSTCN.modules import MBConvBlockV2, FusedMBConvBlockV2, get_activation, round_filters, KeyFeatureExpansionBlock


def round_repeats(repeats, depth_coefficient):
    """Rounds  the number of blocks in a stage to a whole number based
    on depth multiplier described in the original paper."""
    return int(ceil(depth_coefficient * repeats))


class KeyEncoder(nn.Module):

    _models = {'b0': {'num_repeat': [1, 2, 2, 3, 5, 8],
                      'kernel_size': [3, 3, 3, 3, 3, 3],
                      'stride': [1, 2, 2, 2, 1, 2],
                      'expand_ratio': [1, 4, 4, 4, 6, 6],
                      'in_channel': [32, 16, 32, 48, 96, 112],
                      'out_channel': [16, 32, 48, 96, 112, 192],
                      'se_ratio': [None, None, None, 0.25, 0.25, 0.25],
                      'conv_type': [1, 1, 1, 0, 0, 0],
                      'is_feature_stage': [True, True, True, False, True, True],
                      'width_coefficient': 1.0,
                      'depth_coefficient': 1.0,
                      'train_size': 192,
                      'eval_size': 224,
                      'dropout': 0.2,
                      'weight_url': 'https://api.onedrive.com/v1.0/shares/u!aHR0cHM6Ly8xZHJ2Lm1zL3UvcyFBdGlRcHc5VGNjZmlnUVBhWkZRcWNXR3dINmRLP2U9UUI5ZndH/root/content',
                      'model_name': 'efficientnet_v2_b0_21k_ft1k-a91e14c5.pth'},
               'b1': {'num_repeat': [1, 2, 2, 3, 5, 8],
                      'kernel_size': [3, 3, 3, 3, 3, 3],
                      'stride': [1, 2, 2, 2, 1, 2],
                      'expand_ratio': [1, 4, 4, 4, 6, 6],
                      'in_channel': [32, 16, 32, 48, 96, 112],
                      'out_channel': [16, 32, 48, 96, 112, 192],
                      'se_ratio': [None, None, None, 0.25, 0.25, 0.25],
                      'conv_type': [1, 1, 1, 0, 0, 0],
                      'is_feature_stage': [True, True, True, False, True, True],
                      'width_coefficient': 1.0,
                      'depth_coefficient': 1.1,  # 1.1 default
                      'train_size': 192,
                      'eval_size': 240,
                      'dropout': 0.2,
                      'weight_url': 'https://api.onedrive.com/v1.0/shares/u!aHR0cHM6Ly8xZHJ2Lm1zL3UvcyFBdGlRcHc5VGNjZmlnUVJnVGV5UndSY2J2amwtP2U9dTBiV1lO/root/content',
                      'model_name': 'efficientnet_v2_b1_21k_ft1k-58f4fb47.pth'},
               'b2': {'num_repeat': [1, 2, 2, 3, 5, 8],
                      'kernel_size': [3, 3, 3, 3, 3, 3],
                      'stride': [1, 2, 2, 2, 1, 2],
                      'expand_ratio': [1, 4, 4, 4, 6, 6],
                      'in_channel': [32, 16, 32, 48, 96, 112],
                      'out_channel': [16, 32, 48, 96, 112, 192],
                      'se_ratio': [None, None, None, 0.25, 0.25, 0.25],
                      'conv_type': [1, 1, 1, 0, 0, 0],
                      'is_feature_stage': [True, True, True, False, True, True],
                      'width_coefficient': 1.1,
                      'depth_coefficient': 1.2,
                      'train_size': 208,
                      'eval_size': 260,
                      'dropout': 0.3,
                      'weight_url': 'https://api.onedrive.com/v1.0/shares/u!aHR0cHM6Ly8xZHJ2Lm1zL3UvcyFBdGlRcHc5VGNjZmlnUVY4M2NySVFZbU41X0tGP2U9ZERZVmxK/root/content',
                      'model_name': 'efficientnet_v2_b2_21k_ft1k-db4ac0ee.pth'},
               'b3': {'num_repeat': [1, 2, 2, 3, 5, 8],
                      'kernel_size': [3, 3, 3, 3, 3, 3],
                      'stride': [1, 2, 2, 2, 1, 2],
                      'expand_ratio': [1, 4, 4, 4, 6, 6],
                      'in_channel': [32, 16, 32, 48, 96, 112],
                      'out_channel': [16, 32, 48, 96, 112, 192],
                      'se_ratio': [None, None, None, 0.25, 0.25, 0.25],
                      'conv_type': [1, 1, 1, 0, 0, 0],  # 1, 1, 1, 0, 0, 0 Fused = 1
                      'is_feature_stage': [True, True, True, False, True, True],
                      'width_coefficient': 1.2,
                      'depth_coefficient': 1.4,
                      'train_size': 240,
                      'eval_size': 300,
                      'dropout': 0.3,
                      'weight_url': 'https://api.onedrive.com/v1.0/shares/u!aHR0cHM6Ly8xZHJ2Lm1zL3UvcyFBdGlRcHc5VGNjZmlnUVpkamdZUzhhaDdtTTZLP2U9anA4VWN2/root/content',
                      'model_name': 'efficientnet_v2_b3_21k_ft1k-3da5874c.pth'},
               's': {'num_repeat': [2, 4, 4, 6, 9, 15],
                     'kernel_size': [3, 3, 3, 3, 3, 3],
                     'stride': [1, 2, 2, 2, 1, 2],
                     'expand_ratio': [1, 4, 4, 4, 6, 6],
                     'in_channel': [24, 24, 48, 64, 128, 160],
                     'out_channel': [24, 48, 64, 128, 160, 256],
                     'se_ratio': [None, None, None, 0.25, 0.25, 0.25],
                     'conv_type': [1, 1, 1, 0, 0, 0],
                     'is_feature_stage': [True, True, True, False, True, True],
                     'width_coefficient': 1.0,
                     'depth_coefficient': 1.0,
                     'train_size': 300,
                     'eval_size': 384,
                     'dropout': 0.2,
                     'weight_url': 'https://api.onedrive.com/v1.0/shares/u!aHR0cHM6Ly8xZHJ2Lm1zL3UvcyFBdGlRcHc5VGNjZmllbFF5VWJOZzd0cmhBbm8/root/content',
                     'model_name': 'efficientnet_v2_s_21k_ft1k-dbb43f38.pth'},
               'm': {'num_repeat': [3, 5, 5, 7, 14, 18, 5],
                     'kernel_size': [3, 3, 3, 3, 3, 3, 3],
                     'stride': [1, 2, 2, 2, 1, 2, 1],
                     'expand_ratio': [1, 4, 4, 4, 6, 6, 6],
                     'in_channel': [24, 24, 48, 80, 160, 176, 304],
                     'out_channel': [24, 48, 80, 160, 176, 304, 512],
                     'se_ratio': [None, None, None, 0.25, 0.25, 0.25, 0.25],
                     'conv_type': [1, 1, 1, 0, 0, 0, 0],
                     'is_feature_stage': [True, True, True, False, True, True],
                     'width_coefficient': 1.0,
                     'depth_coefficient': 1.0,
                     'train_size': 384,
                     'eval_size': 480,
                     'dropout': 0.3,
                     'weight_url': 'https://api.onedrive.com/v1.0/shares/u!aHR0cHM6Ly8xZHJ2Lm1zL3UvcyFBdGlRcHc5VGNjZmllN1ZDazRFb0o1bnlyNUE/root/content',
                     'model_name': 'efficientnet_v2_m_21k_ft1k-da8e56c0.pth'},
               'l': {'num_repeat': [4, 7, 7, 10, 19, 25, 7],
                     'kernel_size': [3, 3, 3, 3, 3, 3, 3],
                     'stride': [1, 2, 2, 2, 1, 2, 1],
                     'expand_ratio': [1, 4, 4, 4, 6, 6, 6],
                     'in_channel': [32, 32, 64, 96, 192, 224, 384],
                     'out_channel': [32, 64, 96, 192, 224, 384, 640],
                     'se_ratio': [None, None, None, 0.25, 0.25, 0.25, 0.25],
                     'conv_type': [1, 1, 1, 0, 0, 0, 0],
                     'is_feature_stage': [False, True, True, False, True, True],
                     'feature_stages': [1, 2, 4, 6],
                     'width_coefficient': 1.0,
                     'depth_coefficient': 1.0,
                     'train_size': 384,
                     'eval_size': 480,
                     'dropout': 0.4,
                     'weight_url': 'https://api.onedrive.com/v1.0/shares/u!aHR0cHM6Ly8xZHJ2Lm1zL3UvcyFBdGlRcHc5VGNjZmlmcmIyRHEtQTBhUTBhWVE/root/content',
                     'model_name': 'efficientnet_v2_l_21k_ft1k-08121eee.pth'},
               'xl': {'num_repeat': [4, 8, 8, 16, 24, 32, 8],
                      'kernel_size': [3, 3, 3, 3, 3, 3, 3],
                      'stride': [1, 2, 2, 2, 1, 2, 1],
                      'expand_ratio': [1, 4, 4, 4, 6, 6, 6],
                      'in_channel': [32, 32, 64, 96, 192, 256, 512],
                      'out_channel': [32, 64, 96, 192, 256, 512, 640],
                      'se_ratio': [None, None, None, 0.25, 0.25, 0.25, 0.25],
                      'conv_type': [1, 1, 1, 0, 0, 0, 0],
                      'is_feature_stage': [False, True, True, False, True, True],
                      'feature_stages': [1, 2, 4, 6],
                      'width_coefficient': 1.0,
                      'depth_coefficient': 1.0,
                      'train_size': 384,
                      'eval_size': 512,
                      'dropout': 0.4,
                      'weight_url': 'https://api.onedrive.com/v1.0/shares/u!aHR0cHM6Ly8xZHJ2Lm1zL3UvcyFBdGlRcHc5VGNjZmlmVXQtRHJLa21taUkxWkE/root/content',
                      'model_name': 'efficientnet_v2_xl_21k_ft1k-1fcc9744.pth'}}

    def __init__(self,
                 model_name,
                 in_channels=3,
                 in_spatial_shape=None,
                 activation='silu',
                 activation_kwargs=None,
                 bias=False,
                 drop_connect_rate=0.2,
                 dropout_rate=None,
                 bn_epsilon=1e-3,
                 bn_momentum=0.01,
                 pretrained=False,
                 progress=True,
                 width_coefficient=None,
                 depth_coefficient=None,
                 feature_expansion=False,
                 expansion_ch=1024
                 ):
        super().__init__()

        # As the model is built the repeated sections are added to the blocks list
        # self.chls = in_channels
        self.blocks = nn.ModuleList()
        self.feature_expansion = feature_expansion

        # The index of the final block in each is stage is appended to end_of_stage_idxs
        # During forward() these indexes can be used to extract the appropriate features
        # for the decoder/ feature pyramid network FPN of a segmentation model.
        self.end_of_stage_idxs = []
        self.model_name = model_name
        self.cfg = self._models[model_name]

        # Scale the model as required
        if width_coefficient is not None:
            self.cfg['width_coefficient'] = width_coefficient
        if depth_coefficient is not None:
            self.cfg['depth_coefficient'] = depth_coefficient

        activation_kwargs = {} if activation_kwargs is None else activation_kwargs
        dropout_rate = self.cfg['dropout'] if dropout_rate is None else dropout_rate
        _input_ch = in_channels

        # Define the stem block of EffNetV2. The repeated scalable blocks are
        # appended to this

        self.stem_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=round_filters(self.cfg['in_channel'][0], self.cfg['width_coefficient']),
            kernel_size=3,
            stride=2,
            padding=1,
            bias=bias
        )

        self.stem_bn = nn.BatchNorm2d(
            num_features=round_filters(self.cfg['in_channel'][0], self.cfg['width_coefficient']),
            eps=bn_epsilon,
            momentum=bn_momentum)

        self.stem_act = get_activation(activation, **activation_kwargs)
        drop_connect_rates = self.get_dropconnect_rates(drop_connect_rate)
        self.feature_block_ids = []  # Stores the indexes of the end of stage conv stages

        # force stages zip to drop final stage since zip is determined by the shortest element
        if self.model_name == 'm' or self.model_name == 'l' or self.model_name == 'xl':
            self.cfg['out_channel'] = self.cfg['out_channel'][:-2]  # removes last values
        else:
            self.cfg['out_channel'] = self.cfg['out_channel'][:-1]  # removes last value

        # Determine the in & out channels and number of repeated blocks for the model's scaling  coefficients
        self.cfg['in_channel'] = (
            [round_filters(chs, self.cfg['width_coefficient']) for chs in self.cfg['in_channel']])
        self.cfg['out_channel'] = (
            [round_filters(chs, self.cfg['width_coefficient']) for chs in self.cfg['out_channel']])
        self.cfg['num_repeat'] = (
            [round_repeats(rpt, self.cfg['depth_coefficient']) for rpt in self.cfg['num_repeat']])
        print(f"Stage Channels: {self.cfg['out_channel']}")
        # Unpack self.cfg and zip values to build the main blocks/stages of EfficientNet V2
        stages = zip(*[self.cfg[x] for x in
                       ['num_repeat', 'kernel_size', 'stride', 'expand_ratio', 'in_channel', 'out_channel', 'se_ratio',
                        'conv_type', 'is_feature_stage']])
        idx = 0
        # self.cfg['num_repeat'] = self.cfg['num_repeat'][:-2]  # force the zip function to lose the last two stages

        for stage_args in stages:
            (num_repeat, kernel_size, stride, expand_ratio,
             in_channels, out_channels, se_ratio, conv_type, is_feature_stage) = stage_args

            conv_block = MBConvBlockV2 if conv_type == 0 else FusedMBConvBlockV2

            for _ in range(num_repeat):
                se_size = None if se_ratio is None else max(1, int(in_channels * se_ratio))
                _b = conv_block(in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=kernel_size,
                                stride=stride,
                                expansion_factor=expand_ratio,
                                act_fn=activation,
                                act_kwargs=activation_kwargs,
                                bn_epsilon=bn_epsilon,
                                bn_momentum=bn_momentum,
                                se_size=se_size,
                                drop_connect_rate=drop_connect_rates[idx],
                                bias=bias,
                                in_spatial_shape=in_spatial_shape
                                )
                self.blocks.append(_b)
                idx += 1
                in_channels = out_channels
                stride = 1

            if is_feature_stage:
                self.feature_block_ids.append(idx - 1)
                self.end_of_stage_idxs.append(idx - 1)

        if self.feature_expansion:
            self.feature_expansion = KeyFeatureExpansionBlock(self.cfg['out_channel'][-1], expansion_ch)
        self.stage_outputs_dict = OrderedDict()  # fpn requires order dict of stage feature maps

        if pretrained:
            self._load_state(_input_ch, progress)

        return

    def _load_state(self, in_channels, progress):
        state_dict = model_zoo.load_url(self.cfg['weight_url'],
                                        progress=progress,
                                        file_name=self.cfg['model_name'])

        state_dict = OrderedDict(
            [(k.replace('.conv.', '.'), v) if '.conv.' in k else (k, v) for k, v in state_dict.items()])

        if in_channels > 3:
            # if there are extra channels then we use the pretrained weights for first three (RGB) with added random
            # weights for the extra channels for the initial stem convolution layer
            print(state_dict['stem_conv.weight'].shape, self.stem_conv.weight.shape)
            pre_weights_with_extra_ch = torch.cat((state_dict['stem_conv.weight'], self.stem_conv.weight[:, 3:, :]), 1)
            state_dict['stem_conv.weight'] = pre_weights_with_extra_ch


        if in_channels < 3:
            # if the input channels are less than 3 then we do not restore saved weights to the stem convolution layer
            state_dict.pop('stem_conv.weight')

        # we remove the classification weights for the segmentation model
        state_dict.pop('fc.weight')
        state_dict.pop('fc.bias')

        self.load_state_dict(state_dict, strict=False)
        print("Model weights loaded successfully.")

    def get_dropconnect_rates(self, drop_connect_rate):
        nr = self.cfg['num_repeat']
        dc = self.cfg['depth_coefficient']
        total = sum(round_repeats(nr[i], dc) for i in range(len(nr)))
        return [drop_connect_rate * i / total for i in range(total)]

    def get_features(self, x):
        """Given input x:  returns a list of features from the end of each stage
        that has been set as True in the _model dictionary"""
        x = self.stem_act(self.stem_bn(self.stem_conv(x)))

        features = []
        feat_idx = 0
        for block_idx, block in enumerate(self.blocks):
            x = block(x)
            if block_idx == self.feature_block_ids[feat_idx]:
                features.append(x)
                feat_idx += 1

        return features

    # def forward(self, x):
    #     stage_idx = 1
    #     x = self.stem_act(self.stem_bn(self.stem_conv(x)))
    #     for idx, block in enumerate(self.blocks):
    #         x = block(x)
    #         if idx in self.end_of_stage_idxs:
    #             self.stage_outputs_dict['block_' + str(stage_idx)] = x
    #             stage_idx += 1
    #     x = self.fpn(self.stage_outputs_dict)
    #     return self.head_out(self.final_decon(x['block_1']))

    def forward(self, x):
        """ Returns an ordered dictionary of the 4 CNN stage outputs for input to the decoder"""
        stage_idx = 1
        x = self.stem_act(self.stem_bn(self.stem_conv(x)))
        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx in self.end_of_stage_idxs:
                self.stage_outputs_dict['stage_' + str(stage_idx)] = x
                stage_idx += 1
        if self.feature_expansion:
            self.stage_outputs_dict['stage_4'] = self.feature_expansion(self.stage_outputs_dict['stage_4'])
        return self.stage_outputs_dict



