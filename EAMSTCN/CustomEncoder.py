"""
Modified ResNet structure for a custom encoder using MBConv Blocks Instead of Residual Blocks
Used in the ablation experiment to explore a wider final stage for better segmentation.
"""

from collections import OrderedDict
import torch
import torch.nn as nn
from EAMSTCN.modules import DropConnect, get_activation, SqueezeExcitate, \
    torch_conv_out_spatial_shape, MBConvBlockV2, FusedMBConvBlockV2


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


if __name__ == '__main__':
    q = CustomEncoder()
    im = torch.rand(1, 3, 480, 854)
    x = q(im)
    print(type(x))
    for stage in x.values():
        print(stage.shape)
