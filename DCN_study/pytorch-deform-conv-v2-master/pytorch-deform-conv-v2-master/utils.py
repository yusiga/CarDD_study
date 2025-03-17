import argparse
import random
import math
from PIL import Image
import numpy as np

import torch


# 处理命令行参数的 bool 值
def str2bool(v):
    if v.lower() in ['true', 1]:
        return True
    elif v.lower() in ['false', 0]:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


# 计算模型可训练参数的数量
def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# 计算均值
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# 计算 top-k 预测准确率，适用于多分类任务
def accuracy(output, target, topk=(1,)):
    """
    Computes the accuracy over the k top predictions for the specified values of k
    Args:
        output: 形状为 (batch_size, num_classes)，表示模型对每个样本在 num_classes 个类别上的预测得分
        target: 形状为 (batch_size,)，表示每个样本的真实类别索引
        topk: 传入一个元组，表示需要计算 Top-k 准确率，默认为 (1,)，即计算 Top-1 准确率
    """
    with torch.no_grad():
        maxk = max(topk)  # 获取 topk 元组中的最大值，表示计算时需要取的最大 Top-k 预测
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)  # 取出 top-k 个最高分数的预测类别索引
        pred = pred.t()  # 转置，(batch_size, maxk) → (maxk, batch_size)，每一行表示所有样本的同一级预测结果，方便计算正确率
        # view(1, -1) 让 target 从 (batch_size,) → (1, batch_size)
        # expand_as(pred)：(1, batch_size) → (maxk, batch_size)，让 target 复制 maxk 行，使得它的形状与 pred 相同
        # .eq()：逐元素比较 pred 和 target，如果 pred[i][j] == target[j]，则 correct[i][j] = True，否则为 False
        correct = pred.eq(target.view(1, -1).expand_as(pred))  # 比较预测结果与真实值，返回布尔矩阵

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)  # 计算前 k 个预测中正确的数量
            res.append(correct_k.mul_(100.0 / batch_size))  # 计算准确率（百分比）

        # res 是一个包含多个准确率的列表，长度等于 topk。示例：
        # top1_acc, top5_acc = accuracy(output, target, topk=(1, 5))
        # print(top1_acc, top5_acc)
        return res
