# tf_resnet_v1_101.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

_weights_dict = dict()

def load_weights(weight_file):
    if weight_file is None:
        return {}
    try:
        weights_dict = np.load(weight_file, allow_pickle=True).item()
    except:
        weights_dict = np.load(weight_file, allow_pickle=True, encoding='bytes').item()
    return weights_dict


def _conv(name, in_channels, out_channels, kernel_size, stride, bias):
    conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                     stride=stride, padding=kernel_size[0] // 2, bias=bias is not None)
    if name in _weights_dict:
        w = _weights_dict[name + '/weights']
        conv.weight.data.copy_(torch.from_numpy(w))
        if bias is not None:
            b = _weights_dict[name + '/biases']
            conv.bias.data.copy_(torch.from_numpy(b))
    return conv


def _bn(name, num_features):
    bn = nn.BatchNorm2d(num_features)
    if name in _weights_dict:
        bn.weight.data.copy_(torch.from_numpy(_weights_dict[name + '/gamma']))
        bn.bias.data.copy_(torch.from_numpy(_weights_dict[name + '/beta']))
        bn.running_mean.copy_(torch.from_numpy(_weights_dict[name + '/moving_mean']))
        bn.running_var.copy_(torch.from_numpy(_weights_dict[name + '/moving_variance']))
    return bn


def bottleneck_unit(name, in_channels, bottleneck_channels, out_channels, stride, downsample):
    layers = []
    shortcut = nn.Identity()

    if downsample:
        shortcut = nn.Sequential(
            _conv(f'{name}/shortcut/Conv2D', in_channels, out_channels, (1, 1), (stride, stride), bias=True),
            _bn(f'{name}/shortcut/BatchNorm', out_channels)
        )

    layers.append(_conv(f'{name}/conv1/Conv2D', in_channels, bottleneck_channels, (1, 1), (1, 1), bias=None))
    layers.append(_bn(f'{name}/conv1/BatchNorm', bottleneck_channels))
    layers.append(nn.ReLU(inplace=True))

    layers.append(_conv(f'{name}/conv2/Conv2D', bottleneck_channels, bottleneck_channels, (3, 3), (stride, stride), bias=None))
    layers.append(_bn(f'{name}/conv2/BatchNorm', bottleneck_channels))
    layers.append(nn.ReLU(inplace=True))

    layers.append(_conv(f'{name}/conv3/Conv2D', bottleneck_channels, out_channels, (1, 1), (1, 1), bias=True))
    layers.append(_bn(f'{name}/conv3/BatchNorm', out_channels))

    return nn.Sequential(*layers), shortcut


class tf_resnet_v1_101(nn.Module):
    def __init__(self, weight_file=None):
        super().__init__()
        global _weights_dict
        _weights_dict = load_weights(weight_file)

        self.conv1 = _conv('resnet_v1/conv1', 3, 64, (7, 7), (2, 2), bias=True)
        self.bn1 = _bn('resnet_v1/conv1/BatchNorm', 64)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.block1 = self._make_block('resnet_v1/block1', 64, 64, 256, 3, first_stride=1)
        self.block2 = self._make_block('resnet_v1/block2', 256, 128, 512, 4, first_stride=2)
        self.block3 = self._make_block('resnet_v1/block3', 512, 256, 1024, 23, first_stride=2)
        self.block4 = self._make_block('resnet_v1/block4', 1024, 512, 2048, 3, first_stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, 1000)

    def _make_block(self, base_name, in_channels, bottleneck_channels, out_channels, num_units, first_stride):
        blocks = []
        for i in range(num_units):
            unit_name = f'{base_name}/unit_{i + 1}/bottleneck'
            stride = first_stride if i == 0 else 1
            downsample = (i == 0)
            bottleneck, shortcut = bottleneck_unit(unit_name, in_channels if i == 0 else out_channels,
                                                    bottleneck_channels, out_channels, stride, downsample)
            blocks.append(ResidualBottleneck(bottleneck, shortcut))
        return nn.Sequential(*blocks)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class ResidualBottleneck(nn.Module):
    def __init__(self, bottleneck, shortcut):
        super().__init__()
        self.bottleneck = bottleneck
        self.shortcut = shortcut

    def forward(self, x):
        return F.relu(self.bottleneck(x) + self.shortcut(x))

KitModel = tf_resnet_v1_101