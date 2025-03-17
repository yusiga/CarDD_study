# -*- coding: utf-8 -*-
import numpy as np

from torch import nn
from torch.nn import functional as F
import torch
from torchvision import models
import torchvision

import sys
import os

# 获取当前脚本所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取 deform_conv_v2 所在目录（在上一级目录）
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
# 添加到 Python 的搜索路径
sys.path.append(parent_dir)

from deform_conv_v2 import *


# 使用了 DCN v2 的神经网络模型
class ScaledMNISTNet(nn.Module):
    def __init__(self, args, num_classes):
        super().__init__()

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        # 窗口大小 (2,2)，用于降低特征图尺寸，减少计算量。
        self.pool = nn.MaxPool2d((2, 2))
        # 自适应平均池化：将输出特征图缩小为 1×1，适用于不同大小的输入，使后续的全连接层可以处理固定大小的输入。
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        features = []
        inplanes = 1  # 输入通道数，MNIST 数据集是灰度图，所以初始值为 1。
        outplanes = 32  # 第一个卷积层的输出通道数，初始为 32，之后每层翻倍。
        # 一共四个卷积层，每个卷积层后面跟 BN 和 ReLU
        # DCN 出现在 args.min_deform_layer 及之后的层中
        for i in range(4):
            if args.deform and args.min_deform_layer <= i + 1:
                features.append(DeformConv2d(inplanes, outplanes, 3, padding=1, bias=False, modulation=args.modulation))
            else:
                features.append(nn.Conv2d(inplanes, outplanes, 3, padding=1, bias=False))
            features.append(nn.BatchNorm2d(outplanes))
            features.append(self.relu)
            if i == 1:
                features.append(self.pool)  # 在 第二个卷积层后 进行最大池化
            inplanes = outplanes
            outplanes *= 2
        self.features = nn.Sequential(*features)

        self.fc = nn.Linear(256, num_classes)

    def forward(self, input):
        x = self.features(input)
        x = self.avg_pool(x)  # 进行自适应平均池化，输出变为 (batch_size, 256, 1, 1)
        # view() 是 PyTorch 中的一个张量操作方法，用于 改变张量的形状（Reshape），但不改变数据本身。
        # 如果不确定某个维度的大小，可以用 -1 让 PyTorch 自动计算
        x = x.view(x.shape[0], -1)  # 将 (batch_size, 256, 1, 1) 变成 (batch_size, 256)
        output = self.fc(x)  # 通过全连接层得到最终分类输出

        return output
