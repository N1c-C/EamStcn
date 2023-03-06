"""
STCN: GitHub code for the encoders. Modified to fit EAMSTCN producing an OrderedDict() of the stage outputs
"""

import torch.nn.functional as F
from torchvision import models
from collections import OrderedDict
import math
import torch
import torch.nn as nn
from torch.utils import model_zoo
from EAMSTCN import *
from EAMSTCN.modules import DropConnect, get_activation, SqueezeExcitate, torch_conv_out_spatial_shape


class FusedMBConvBlockV2(nn.Module):
    expansion = 4

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 expansion_factor=1.0,
                 act_fn='swish',
                 act_kwargs=None,
                 bn_epsilon=0.001,
                 bn_momentum=0.01,
                 se_size=None,
                 drop_connect_rate=0.2,
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


class CustomEncoder(nn.Module):

    def __init__(self, stages=(3, 4, 3, 3), dim=(64, 128, 256, 512)):  # [3, 4, 6, 3] 64, 256, 512, 1024
        super().__init__()
        print(stages, dim)
        self.conv1 = nn.Conv2d(3, dim[0], kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(dim[0])
        self.relu = nn.ReLU(inplace=True)  # 1/2 dim[0]
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        layer1 = nn.ModuleList()
        for i in range(stages[0]):
            layer1.append(FusedMBConvBlockV2(dim[0], dim[0], 3, 1, 1, 'silu',
                                             drop_connect_rate=0.2, bn_epsilon=1e-3, bn_momentum=0.01))
        self.layer1 = nn.Sequential(*layer1)

        layer2 = nn.ModuleList()
        layer2.append(FusedMBConvBlockV2(dim[0], dim[1], 3, 1, 1, 'silu',
                                         drop_connect_rate=0.2, bn_epsilon=1e-3, bn_momentum=0.01))
        for i in range(stages[1] - 1):
            layer2.append(FusedMBConvBlockV2(dim[1], dim[1], 3, 1, 1, 'silu',
                                             drop_connect_rate=0.2, bn_epsilon=1e-3, bn_momentum=0.01))
        self.layer2 = nn.Sequential(*layer2)

        layer3 = nn.ModuleList()
        layer3.append(FusedMBConvBlockV2(dim[1], dim[2], 3, 2, 1, 'silu',
                                         drop_connect_rate=0.2, bn_epsilon=1e-3, bn_momentum=0.01))
        for i in range(stages[2] - 1):
            layer3.append(FusedMBConvBlockV2(dim[2], dim[2], 3, 1, 1, 'silu',
                                             drop_connect_rate=0.2, bn_epsilon=1e-3, bn_momentum=0.01))
        self.layer3 = nn.Sequential(*layer3)

        layer4 = nn.ModuleList()
        layer4.append(MBConvBlockV2(dim[2], dim[3], 3, 2, 1, 'silu',
                                    drop_connect_rate=0.2, bn_epsilon=1e-3, bn_momentum=0.01))
        for i in range(stages[3] - 1):
            layer4.append(MBConvBlockV2(dim[3], dim[3], 3, 1, 1, 'silu',
                                        drop_connect_rate=0.2, bn_epsilon=1e-3, bn_momentum=0.01))
        self.layer4 = nn.Sequential(*layer4)

    def forward(self, f):
        features = OrderedDict()
        x = self.conv1(f)
        x = self.bn1(x)
        x = self.relu(x)  # 1/2, dim 0
        x = self.layer1(x)
        features['stage_1'] = x
        x = self.maxpool(x)  # 1/4, dim 0
        x = self.layer2(x)  # 1/4, dim 1
        features['stage_2'] = x
        x = self.layer3(x)  # 1/8, dim 2
        features['stage_3'] = x
        x = self.layer4(x)  # 1/16, dim 3
        features['stage_4'] = x
        return features


def load_weights_sequential(target, source_state, extra_chan=1):
    new_dict = OrderedDict()

    for k1, v1 in target.state_dict().items():
        if not 'num_batches_tracked' in k1:
            if k1 in source_state:
                tar_v = source_state[k1]

                if v1.shape != tar_v.shape:
                    # Init the new segmentation channel with zeros
                    # print(v1.shape, tar_v.shape)
                    c, _, w, h = v1.shape
                    pads = torch.zeros((c, extra_chan, w, h), device=tar_v.device)
                    nn.init.orthogonal_(pads)
                    tar_v = torch.cat([tar_v, pads], 1)

                new_dict[k1] = tar_v

    target.load_state_dict(new_dict, strict=False)


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
}


def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, dilation=dilation)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride=stride, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, stride=1, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class MBConvBlockV2(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 expansion_factor=1.0,
                 act_fn='swish',
                 act_kwargs=None,
                 bn_epsilon=None,
                 bn_momentum=None,
                 se_size=None,
                 drop_connect_rate=0.2,
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


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, dilation=dilation,
                               padding=dilation)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers=(3, 4, 23, 3), extra_chan=1):
        self.inplanes = 64
        super().__init__()
        self.conv1 = nn.Conv2d(3 + extra_chan, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)


def resnet18(pretrained=True, extra_chan=0):
    model = ResNet(BasicBlock, [2, 2, 2, 2], extra_chan)
    if pretrained:
        load_weights_sequential(model, model_zoo.load_url(model_urls['resnet18']), extra_chan)
    return model


def resnet50(pretrained=True, extra_chan=0):
    model = ResNet(Bottleneck, [3, 4, 6, 3], extra_chan)
    return model


if __name__ == '__main__':
    q = CustomEncoder()
    im = torch.rand(1, 3, 480, 854)
    x = q(im)
    print(type(x))
    for stage in x.values():
        print(stage.shape)
