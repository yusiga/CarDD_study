import os
import argparse
import numpy as np
from tqdm import tqdm
import pandas as pd
import joblib
from collections import OrderedDict
from datetime import datetime

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from utils import *
from scaled_mnist.dataset import ScaledMNIST
import scaled_mnist.archs as archs

arch_names = archs.__dict__.keys()


def parse_args():
    # 使用 argparse 解析命令行参数，允许用户在运行脚本时自定义超参数。
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='ScaledMNISTNet',
                        choices=arch_names,
                        help='model architecture: ' +
                             ' | '.join(arch_names) +
                             ' (default: ScaledMNISTNet)')
    parser.add_argument('--deform', default=True, type=str2bool,
                        help='use deform conv')
    parser.add_argument('--modulation', default=True, type=str2bool,
                        help='use modulated deform conv')
    parser.add_argument('--min-deform-layer', default=3, type=int,
                        help='minimum number of layer using deform conv')
    parser.add_argument('--epochs', default=10, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--optimizer', default='SGD',
                        choices=['Adam', 'SGD'],
                        help='loss: ' +
                             ' | '.join(['Adam', 'SGD']) +
                             ' (default: Adam)')
    parser.add_argument('--lr', '--learning-rate', default=1e-2, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.5, type=float,
                        help='momentum')
    parser.add_argument('--weight-decay', default=1e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', default=False, type=str2bool,
                        help='nesterov')

    args = parser.parse_args()

    return args


# 训练函数
def train(args, train_loader, model, criterion, optimizer, epoch, scheduler=None):
    # AverageMeter() 用于计算 损失 和 准确率 的平均值。
    losses = AverageMeter()
    scores = AverageMeter()

    model.train()

    # 使用 tqdm 显示训练进度。
    for i, (input, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
        # 将数据移动到 GPU
        input = input.cuda()
        target = target.cuda()

        # 前向传播 计算输出 output，并计算 loss（损失）。
        output = model(input)
        loss = criterion(output, target)

        # 计算准确率
        acc = accuracy(output, target)[0]

        # 更新 损失 和 准确率 的平均值。
        losses.update(loss.item(), input.size(0))
        scores.update(acc.item(), input.size(0))

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()  # 更新权重

    # 返回模型训练日志
    log = OrderedDict([
        ('loss', losses.avg),
        ('acc', scores.avg),
    ])

    return log


# 验证函数
def validate(args, val_loader, model, criterion):
    losses = AverageMeter()
    scores = AverageMeter()

    # switch to evaluate mode
    model.eval()

    # 不会反向传播
    with torch.no_grad():
        for i, (input, target) in tqdm(enumerate(val_loader), total=len(val_loader)):
            input = input.cuda()
            target = target.cuda()

            output = model(input)
            loss = criterion(output, target)

            acc = accuracy(output, target)[0]

            losses.update(loss.item(), input.size(0))
            scores.update(acc.item(), input.size(0))

    # 返回模型验证日志
    log = OrderedDict([
        ('loss', losses.avg),
        ('acc', scores.avg),
    ])

    return log


# 模型训练的主函数
def main():
    args = parse_args()

    # 自动生成模型名称
    if args.name is None:
        args.name = '%s' % args.arch
        if args.deform:
            args.name += '_wDCN'
            if args.modulation:
                args.name += 'v2'
            args.name += '_c%d-4' % args.min_deform_layer

    # 创建模型保存的地址
    if not os.path.exists('models/%s' % args.name):
        os.makedirs('models/%s' % args.name)

    # 打印实验配置
    print('Config -----')
    for arg in vars(args):
        print('%s: %s' % (arg, getattr(args, arg)))
    print('------------')

    # 输出配置文件
    with open('models/%s/args.txt' % args.name, 'w') as f:
        for arg in vars(args):
            print('%s: %s' % (arg, getattr(args, arg)), file=f)

    # 使用 joblib 将 args 对象序列化并存入 args.pkl 文件。
    # 以后可以用 joblib.load('models/experiment1/args.pkl') 直接加载参数，而无需重新解析命令行参数。
    joblib.dump(args, 'models/%s/args.pkl' % args.name)

    # 定义损失函数 并启用 cudnn.benchmark（提高 CNN 计算速度）。
    criterion = nn.CrossEntropyLoss().cuda()
    cudnn.benchmark = True

    # 加载数据
    # 将 PIL 图片或 NumPy 数组转换为 PyTorch Tensor，并对像素进行归一化
    # 然后对图像进行标准化
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # 加载数据
    train_set = ScaledMNIST(
        train=True,
        transform=transform_train)
    # 批量读取数据
    # 每次加载 32 个样本，随机打乱数据，并使用 8 个子进程加载数据
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=32,
        shuffle=True,
        num_workers=8)

    test_set = ScaledMNIST(
        train=False,
        transform=transform_train)
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=32,
        shuffle=False,
        num_workers=8)

    num_classes = 10

    # 初始化模型并移动到 GPU
    model = archs.__dict__[args.arch](args, num_classes)
    model = model.cuda()

    print(model)

    # 定义优化器
    if args.optimizer == 'Adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    elif args.optimizer == 'SGD':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
                              momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)

    log = pd.DataFrame(index=[], columns=[
        'epoch', 'lr', 'loss', 'acc', 'val_loss', 'val_acc'
    ])

    best_acc = 0
    for epoch in range(args.epochs):
        print('Epoch [%d/%d]' % (epoch, args.epochs))

        # train for one epoch
        train_log = train(args, train_loader, model, criterion, optimizer, epoch)
        # evaluate on validation set
        val_log = validate(args, test_loader, model, criterion)

        print('loss %.4f - acc %.4f - val_loss %.4f - val_acc %.4f'
              % (train_log['loss'], train_log['acc'], val_log['loss'], val_log['acc']))

        tmp = pd.Series([
            epoch,
            1e-1,
            train_log['loss'],
            train_log['acc'],
            val_log['loss'],
            val_log['acc'],
        ], index=['epoch', 'lr', 'loss', 'acc', 'val_loss', 'val_acc'])

        # ignore_index=True 确保 log 仍然是连续的行索引
        log = log.pd.concat([log, tmp], ignore_index=True)
        # 这行代码把 log 保存为 CSV 文件（log.csv）
        # index=False 表示不保存行索引，只存数据本身
        log.to_csv('models/%s/log.csv' % args.name, index=False)

        # 如果当前 epoch 的验证集准确率更高，则保存最佳模型
        if val_log['acc'] > best_acc:
            # 保存当前模型参数
            torch.save(model.state_dict(), 'models/%s/model.pth' % args.name)
            best_acc = val_log['acc']
            print("=> saved best model")

    print("best val_acc: %f" % best_acc)


if __name__ == '__main__':
    main()
