import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ResNetV1_152(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.params = {}
        self.num_classes = num_classes

        # Stem
        self.__conv('conv1', 3, 64, kernel_size=7, stride=2, padding=3)
        self.__bn('conv1')

        # Blocks
        self.__res_block('block1', 64, 64, 3, stride=2)
        self.__res_block('block2', 256, 128, 8, stride=2)
        self.__res_block('block3', 512, 256, 36, stride=2)
        self.__res_block('block4', 1024, 512, 3, stride=1)

        # Final layers
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_classes)

    def __conv(self, name, in_channels, out_channels, kernel_size, stride, padding):
        conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
        setattr(self, name, conv)
        self.params[f'resnet_v1_152/{name}/weights'] = conv.weight

    def __bn(self, name, num_features=None):
        bn = nn.BatchNorm2d(num_features if num_features else getattr(self, name).out_channels)
        setattr(self, name + '_bn', bn)
        self.params[f'resnet_v1_152/{name}/BatchNorm/gamma'] = bn.weight
        self.params[f'resnet_v1_152/{name}/BatchNorm/beta'] = bn.bias
        self.params[f'resnet_v1_152/{name}/BatchNorm/moving_mean'] = bn.running_mean
        self.params[f'resnet_v1_152/{name}/BatchNorm/moving_variance'] = bn.running_var

    def __bottleneck(self, prefix, in_channels, bottleneck_channels, stride):
        out_channels = bottleneck_channels * 4
        shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            shortcut.add_module('conv', nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False))
            shortcut.add_module('bn', nn.BatchNorm2d(out_channels))

        block = nn.Sequential(
            nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(bottleneck_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(bottleneck_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(bottleneck_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        return nn.Sequential(block, shortcut)

    def __res_block(self, name, in_channels, bottleneck_channels, blocks, stride):
        for i in range(blocks):
            unit_name = f'{name}/unit_{i + 1}/bottleneck_v1'
            s = stride if i == blocks - 1 else 1
            block = self.__bottleneck(unit_name, in_channels if i == 0 else bottleneck_channels * 4,
                                      bottleneck_channels, s)
            setattr(self, unit_name.replace('/', '_'), block)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv1_bn(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)

        for stage in ['block1', 'block2', 'block3', 'block4']:
            i = 1
            while hasattr(self, f'{stage}_unit_{i}_bottleneck_v1'):
                block = getattr(self, f'{stage}_unit_{i}_bottleneck_v1')
                x = block[0](x) + (block[1](x) if len(block[1]) > 0 else x)
                x = F.relu(x)
                i += 1

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def load_weights_from_npy(self, npy_path):
        weights = np.load(npy_path, allow_pickle=True).item()
        for name, param in self.params.items():
            if name in weights:
                w = torch.from_numpy(weights[name])
                if param.shape != w.shape:
                    print(f"Shape mismatch: {name}, expected {param.shape}, got {w.shape}")
                else:
                    param.data.copy_(w)
            else:
                print(f"Missing weight: {name}")


class KitModel(nn.Module):
    def __init__(self, npy_path=None):
        super().__init__()
        self.model = ResNetV1_152()
        if npy_path is not None:
            self.model.load_weights_from_npy(npy_path)

    def forward(self, x):
        return self.model(x)