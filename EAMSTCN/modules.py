from collections import abc as container_abc
from math import ceil, floor
import torch
from torch import nn as nn
from torch.nn import functional as F


# Blocks and functions required for EfficientNetV2 https://github.com/abhuse/pytorch-efficientnet
def _pair(x):
    if isinstance(x, container_abc.Iterable):
        return x
    return x, x


def torch_conv_out_spatial_shape(in_spatial_shape, kernel_size, stride):
    if in_spatial_shape is None:
        return None
    # in_spatial_shape -> [H,W]
    hin, win = _pair(in_spatial_shape)
    kh, kw = _pair(kernel_size)
    sh, sw = _pair(stride)

    # dilation and padding are ignored since they are always fixed in efficientnetV2
    hout = int(floor((hin - kh - 1) / sh + 1))
    wout = int(floor((win - kw - 1) / sw + 1))
    return hout, wout


def get_activation(act_fn: str, **kwargs):
    """Returns the  torch activation function given an argument.
    Add/Delete extra activations as neccessary.
    No checking performed on kwargs so check torch documentation for acceptable
    arguments """
    if act_fn in ('silu', 'swish'):
        return nn.SiLU(**kwargs)
    elif act_fn == 'relu':
        return nn.ReLU(**kwargs)
    elif act_fn == 'relu6':
        return nn.ReLU6(**kwargs)
    elif act_fn == 'elu':
        return nn.ELU(**kwargs)
    elif act_fn == 'leaky_relu':
        return nn.LeakyReLU(**kwargs)
    elif act_fn == 'selu':
        return nn.SELU(**kwargs)
    elif act_fn == 'mish':
        return nn.Mish(**kwargs)
    else:
        raise ValueError('Unsupported act_fn {}'.format(act_fn))


def round_filters(filters, width_coefficient, depth_divisor=8):
    """Rounds the number of filters to a whole number after using the appropriate
     depth multiplier for a given model - See paper for the coefficient values ."""
    min_depth = depth_divisor
    filters *= width_coefficient
    new_filters = max(min_depth, int(filters + depth_divisor / 2) // depth_divisor * depth_divisor)
    return int(new_filters)


class DropConnect(nn.Module):
    def __init__(self, rate=0.5):
        super(DropConnect, self).__init__()
        self.keep_prob = None
        self.set_rate(rate)

    def set_rate(self, rate):
        if not 0 <= rate < 1:
            raise ValueError("rate must be 0<=rate<1, got {} instead".format(rate))
        self.keep_prob = 1 - rate

    def forward(self, x):
        if self.training:
            random_tensor = self.keep_prob + torch.rand([x.size(0), 1, 1, 1],
                                                        dtype=x.dtype,
                                                        device=x.device)
            binary_tensor = torch.floor(random_tensor)
            return torch.mul(torch.div(x, self.keep_prob), binary_tensor)
        else:
            return x


class SamePaddingConv2d(nn.Module):
    def __init__(self,
                 in_spatial_shape,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 dilation=1,
                 enforce_in_spatial_shape=False,
                 **kwargs):
        super(SamePaddingConv2d, self).__init__()

        self._in_spatial_shape = _pair(in_spatial_shape)
        # e.g. throw exception if input spatial shape does not match in_spatial_shape
        # when calling self.forward()
        self.enforce_in_spatial_shape = enforce_in_spatial_shape
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        dilation = _pair(dilation)

        in_height, in_width = self._in_spatial_shape
        filter_height, filter_width = kernel_size
        stride_height, stride_width = stride
        dilation_height, dilation_width = dilation

        out_height = int(ceil(float(in_height) / float(stride_height)))
        out_width = int(ceil(float(in_width) / float(stride_width)))

        pad_along_height = max((out_height - 1) * stride_height +
                               filter_height + (filter_height - 1) * (dilation_height - 1) - in_height, 0)
        pad_along_width = max((out_width - 1) * stride_width +
                              filter_width + (filter_width - 1) * (dilation_width - 1) - in_width, 0)

        pad_top = pad_along_height // 2
        pad_bottom = pad_along_height - pad_top
        pad_left = pad_along_width // 2
        pad_right = pad_along_width - pad_left

        paddings = (pad_left, pad_right, pad_top, pad_bottom)
        if any(p > 0 for p in paddings):
            self.zero_pad = nn.ZeroPad2d(paddings)
        else:
            self.zero_pad = None
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              dilation=dilation,
                              **kwargs)

        self._out_spatial_shape = (out_height, out_width)

    @property
    def out_spatial_shape(self):
        return self._out_spatial_shape

    def check_spatial_shape(self, x):
        if x.size(2) != self._in_spatial_shape[0] or \
                x.size(3) != self._in_spatial_shape[1]:
            raise ValueError(
                "Expected input spatial shape {}, got {} instead".format(self._in_spatial_shape, x.shape[2:]))

    def forward(self, x):
        if self.enforce_in_spatial_shape:
            self.check_spatial_shape(x)
        if self.zero_pad is not None:
            x = self.zero_pad(x)
        x = self.conv(x)
        return x


class SqueezeExcitate(nn.Module):
    """Squeeze Excite takes into account the most relevant channels when
    computing the output of a stack of features. Each channel is squeezed  down
    to a single number producing vector N. After some function (activation or small NN)
    the resultant vector acts as a set of weights for the importance of the original features. """

    def __init__(self,
                 in_channels,
                 se_size,
                 activation=None):
        super(SqueezeExcitate, self).__init__()
        self.dim_reduce = nn.Conv2d(in_channels=in_channels,
                                    out_channels=se_size,
                                    kernel_size=1)
        self.dim_restore = nn.Conv2d(in_channels=se_size,
                                     out_channels=in_channels,
                                     kernel_size=1)
        self.activation = F.relu if activation is None else activation

    def forward(self, x):
        inp = x
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = self.dim_reduce(x)
        x = self.activation(x)
        x = self.dim_restore(x)
        x = torch.sigmoid(x)
        return torch.mul(inp, x)


class MBConvBlockV2(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 expansion_factor,
                 act_fn,
                 act_kwargs=None,
                 bn_epsilon=None,
                 bn_momentum=None,
                 se_size=None,
                 drop_connect_rate=None,
                 bias=False,
                 in_spatial_shape=None):

        super().__init__()

        if act_kwargs is None:
            act_kwargs = {}
        exp_channels = in_channels * expansion_factor

        self.ops_lst = []

        # expansion convolution
        if expansion_factor != 1:
            self.expand_conv = nn.Conv2d(in_channels=in_channels,
                                         out_channels=exp_channels,
                                         kernel_size=1,
                                         bias=bias)

            self.expand_bn = nn.BatchNorm2d(num_features=exp_channels,
                                            eps=bn_epsilon,
                                            momentum=bn_momentum)

            self.expand_act = get_activation(act_fn, **act_kwargs)
            self.ops_lst.extend([self.expand_conv, self.expand_bn, self.expand_act])

        # depth-wise convolution
        self.dp_conv = nn.Conv2d(in_channels=exp_channels,
                                 out_channels=exp_channels,
                                 kernel_size=kernel_size,
                                 stride=stride,
                                 padding=1,
                                 groups=exp_channels,
                                 bias=bias)
        self.out_spatial_shape = torch_conv_out_spatial_shape(in_spatial_shape, kernel_size, stride)

        self.dp_bn = nn.BatchNorm2d(num_features=exp_channels,
                                    eps=bn_epsilon,
                                    momentum=bn_momentum)

        self.dp_act = get_activation(act_fn, **act_kwargs)
        self.ops_lst.extend([self.dp_conv, self.dp_bn, self.dp_act])

        # Squeeze and Excitate
        if se_size is not None:
            self.se = SqueezeExcitate(exp_channels,
                                      se_size,
                                      activation=get_activation(act_fn, **act_kwargs))
            self.ops_lst.append(self.se)

        # projection layer
        self.project_conv = nn.Conv2d(in_channels=exp_channels,
                                      out_channels=out_channels,
                                      kernel_size=1,
                                      bias=bias)

        self.project_bn = nn.BatchNorm2d(num_features=out_channels,
                                         eps=bn_epsilon,
                                         momentum=bn_momentum)

        # no activation function in projection layer

        self.ops_lst.extend([self.project_conv, self.project_bn])

        self.skip_enabled = in_channels == out_channels and stride == 1

        if self.skip_enabled and drop_connect_rate is not None:
            self.drop_connect = DropConnect(drop_connect_rate)
            self.ops_lst.append(self.drop_connect)

    def forward(self, x):
        inp = x
        for op in self.ops_lst:
            x = op(x)
        if self.skip_enabled:
            return x + inp
        else:
            return x


class FusedMBConvBlockV2(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 expansion_factor,
                 act_fn,
                 act_kwargs=None,
                 bn_epsilon=None,
                 bn_momentum=None,
                 se_size=None,
                 drop_connect_rate=None,
                 bias=False,
                 padding=1,
                 in_spatial_shape=None):

        super().__init__()

        if act_kwargs is None:
            act_kwargs = {}
        exp_channels = in_channels * expansion_factor

        self.ops_lst = []

        # expansion convolution
        expansion_out_shape = in_spatial_shape
        if expansion_factor != 1:
            self.expand_conv = nn.Conv2d(in_channels=in_channels,
                                         out_channels=exp_channels,
                                         kernel_size=kernel_size,
                                         padding=padding,
                                         stride=stride,
                                         bias=bias)
            expansion_out_shape = torch_conv_out_spatial_shape(in_spatial_shape, kernel_size, stride)

            self.expand_bn = nn.BatchNorm2d(num_features=exp_channels,
                                            eps=bn_epsilon,
                                            momentum=bn_momentum)

            self.expand_act = get_activation(act_fn, **act_kwargs)
            self.ops_lst.extend([self.expand_conv, self.expand_bn, self.expand_act])

        # Squeeze and Excite
        if se_size is not None:
            self.se = SqueezeExcitate(exp_channels,
                                      se_size,
                                      activation=get_activation(act_fn, **act_kwargs))
            self.ops_lst.append(self.se)

        # projection layer
        kernel_size = 1 if expansion_factor != 1 else kernel_size
        stride = 1 if expansion_factor != 1 else stride

        self.project_conv = nn.Conv2d(in_channels=exp_channels,
                                      out_channels=out_channels,
                                      kernel_size=kernel_size,
                                      stride=stride,
                                      padding=1 if kernel_size > 1 else 0,
                                      bias=bias)
        self.out_spatial_shape = torch_conv_out_spatial_shape(expansion_out_shape, kernel_size, stride)

        self.project_bn = nn.BatchNorm2d(num_features=out_channels,
                                         eps=bn_epsilon,
                                         momentum=bn_momentum)

        self.ops_lst.extend(
            [self.project_conv, self.project_bn])

        if expansion_factor == 1:
            self.project_act = get_activation(act_fn, **act_kwargs)
            self.ops_lst.append(self.project_act)

        self.skip_enabled = in_channels == out_channels and stride == 1

        if self.skip_enabled and drop_connect_rate is not None:
            self.drop_connect = DropConnect(drop_connect_rate)
            self.ops_lst.append(self.drop_connect)

    def forward(self, x):
        inp = x
        for op in self.ops_lst:
            x = op(x)
        if self.skip_enabled:
            return x + inp
        else:
            return x


#  CBAM Modified from https://github.com/Jongchan/attention-module/blob/master/MODELS/cbam.py

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super().__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)

    def forward(self, x):
        x = self.conv(x)
        return x


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super().__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(max_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class SpatialGate(nn.Module):
    def __init__(self):
        super().__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)  # broadcasting
        return x * scale


class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super().__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()

    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out


class ResBlock(nn.Module):
    def __init__(self, indim, outdim=None):
        super(ResBlock, self).__init__()
        if outdim == None:
            outdim = indim
        if indim == outdim:
            self.downsample = None
        else:
            self.downsample = nn.Conv2d(indim, outdim, kernel_size=3, padding=1)

        self.conv1 = nn.Conv2d(indim, outdim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(outdim, outdim, kernel_size=3, padding=1)

    def forward(self, x):
        r = self.conv1(F.relu(x))
        r = self.conv2(F.relu(r))

        if self.downsample is not None:
            x = self.downsample(x)

        return x + r


class UpsampleBlock(nn.Module):
    def __init__(self, skip_c, up_c, out_c, scale_factor=2):
        super().__init__()
        self.skip_conv = nn.Conv2d(skip_c, up_c, kernel_size=3, padding=1)
        # self.out_conv = FusedMBConvBlockV2(up_c, out_c, 3, 1, 1, 'silu',
        #                               drop_connect_rate=0.2,
        #                               bn_epsilon=1e-3,
        #                               # se_size=512,
        #                               bn_momentum=0.01)
        self.out_conv = ResBlock(up_c, out_c)
        self.scale_factor = scale_factor

    def forward(self, skip_f, up_f):
        x = self.skip_conv(skip_f)
        x = x + F.interpolate(up_f, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
        x = self.out_conv(x)
        return x


class KeyFeatureExpansionBlock(nn.Module):
    """Block to expand the feature set from the query/key encoder to increase the size
    of the correspondence mapping space. Takes the output from stage_4 of the efficient net as input.
    """

    def __init__(self, in_channel, out_channel):
        super().__init__()

        self.block1 = FusedMBConvBlockV2(in_channel, out_channel, 3, 1, 1, 'silu',
                                         drop_connect_rate=0.2,
                                         bn_epsilon=1e-3,
                                         bn_momentum=0.01)

        self.block2 = MBConvBlockV2(out_channel, out_channel, 3, 1, 1, 'silu',
                                    drop_connect_rate=0.2,
                                    bn_epsilon=1e-3,
                                    bn_momentum=0.01)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        return x


# Feature fusion Modified from https://github.com/hkchengrex/STCN
class FeatureFusionBlock(nn.Module):
    """In the STCN and STM papers ResNet blocks are implemented with CBAM for the feature fusion.
     To maintain the efficientNet architecture Fused Mobile Bottleneck convolution is used instead.
    The final stages from the two encoders are concatenated together to maximise the use of the deeper key network.
    :Returns fused feature map from key encoder stage 4 (f16) and value encoder stage 4 outputs"""

    def __init__(self, in_channel, out_channel, ):
        super().__init__()
        #
        self.block1 = FusedMBConvBlockV2(in_channel, out_channel, 3, 1, 1, 'silu',
                                    drop_connect_rate=0.2,
                                    bn_epsilon=1e-3,
                                    # se_size=256,
                                    bn_momentum=0.01)
        # self.attention = CBAM(out_channel)
        self.block2 = MBConvBlockV2(out_channel, out_channel, 3, 1, 1, 'silu',
                                    drop_connect_rate=0.2,
                                    bn_epsilon=1e-3,
                                    # se_size=256,
                                    bn_momentum=0.01
                                    )
        #
        # self.block1 = ResBlock(in_channel, out_channel)
        # # self.attention = CBAM(out_channel)
        # self.block2 = ResBlock(out_channel, out_channel)

    def forward(self, value_key, key_f16_features):
        x = torch.cat([value_key, key_f16_features], 1)
        # x = self.block1(x)
        # r = self.attention(x)
        # x = self.block2(x + r)
        x = self.block1(x)
        x = self.block2(x)
        return x


class ProjectToKey(nn.Module):
    """ Key projection takes the final conv layer out from the key encoder (query) and projects the final key with
    Ck dimensions (out_channels).
    Ck is set as 128 for STM but STCN demonstrates better performance when set to 64 """

    def __init__(self, in_channel, out_channel):
        """Initialisation follows: Exact solutions to the nonlinear dynamics of learning in deep linear neural networks
        Saxe, A. et al. (2013)  NB: an orthogonal matrix's transpose is equal to its inverse A.AT = AT.A = I
         Bias values set to zero and no non linerarity as per STCN / STM papers"""
        super().__init__()
        self.key_proj = nn.Conv2d(in_channel, out_channel, kernel_size=3, padding='same')

        nn.init.orthogonal_(self.key_proj.weight.data)
        nn.init.zeros_(self.key_proj.bias.data)

    def forward(self, x):
        return self.key_proj(x)
