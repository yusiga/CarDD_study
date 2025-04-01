import os
from PIL import Image
import cv2
import torch
from torch.utils import data
from torchvision import transforms
from torchvision.transforms import functional as F
import numbers
import numpy as np
import random


class ImageDataTrain(data.Dataset):
    def __init__(self, data_root, data_list):
        self.sal_root = data_root
        self.sal_source = data_list

        self.img_list = os.listdir(os.path.join(self.sal_root, 'CarDD-TR-Image'))
        self.gt_list = os.listdir(os.path.join(self.sal_root, 'CarDD-TR-Mask'))

        self.img_list.sort()
        self.gt_list.sort()
        print(self.img_list[0:10])
        print(self.gt_list[0:10])

        #        with open(self.sal_source, 'r') as f:
        #            self.sal_list = [x.strip() for x in f.readlines()]

        self.sal_num = len(self.img_list)

    def __getitem__(self, item):
        # sal data loading
        im_name = self.img_list[item % self.sal_num]
        gt_name = self.gt_list[item % self.sal_num]

        # load_image() 归一化预处理
        # load_sal_label() 读取掩码图，并转换为 0-1 之间的浮点数
        sal_image = load_image(os.path.join(self.sal_root, 'CarDD-TR-Image', im_name))
        sal_label = load_sal_label(os.path.join(self.sal_root, 'CarDD-TR-Mask', gt_name))
        # sal_edge = load_sal_label(os.path.join(self.sal_root,'CarDD-TR-Edge' ,gt_name.replace('.png','_edge.png')))
        sal_edge = load_sal_label(os.path.join(self.sal_root, 'CarDD-TR-Edge', gt_name))

        # 数据增强
        # 1.随机水平翻转
        sal_image, sal_label, sal_edge = cv_random_flip(sal_image, sal_label, sal_edge)
        # 2.转换通道顺序（HWC → CHW）
        sal_image = sal_image.transpose((1, 2, 0))
        sal_label = sal_label.transpose((1, 2, 0))
        sal_edge = sal_edge.transpose((1, 2, 0))
        # resize 会导致后续层 input 维度为 0 的错误，所以这里注释掉了
        # sal_image, sal_label, sal_edge = generate_scale_label(sal_image, sal_label, sal_edge)
        # 3.随机旋转图像（角度 -25° ~ 25°）
        sal_image, sal_label, sal_edge = random_rotate(sal_image, sal_label, sal_edge)
        # 4.还原通道顺序
        sal_image = sal_image.transpose((2, 0, 1))
        sal_label = sal_label.transpose((2, 0, 1))
        sal_edge = sal_edge.transpose((2, 0, 1))
        # 5.将二值掩码图转换为显著性图
        # 对 sal_label 进行高斯模糊，得到显著性图 sal_saliency。
        # squeeze 去除大小为 1 的维度，可以指定（axis=None），如果不指定，则删除全部大小为 1 的
        sal_saliency = np.squeeze(sal_label)
        kernel_size = (25, 25)
        sal_saliency = cv2.GaussianBlur(sal_saliency, kernel_size, 8)  # 平滑 sal_saliency，减少噪声，使其更加平滑。
        sal_saliency = sal_saliency / np.max(sal_saliency)  # 归一化
        sal_saliency = np.expand_dims(sal_saliency, axis=0)  # 扩展维度，适合后面的批量处理
        # 转换为 PyTorch Tensor
        sal_image = torch.Tensor(sal_image)
        sal_label = torch.Tensor(sal_label)
        sal_edge = torch.Tensor(sal_edge)
        sal_saliency = torch.Tensor(sal_saliency)

        # 以字典的形式返回样本
        sample = {'sal_image': sal_image, 'sal_label': sal_label, 'sal_edge': sal_edge, 'sal_saliency': sal_saliency,
                  'im_name': im_name}
        return sample

    def __len__(self):
        return self.sal_num


class ImageDataTest(data.Dataset):
    def __init__(self, data_root, data_list):
        self.data_root = data_root
        self.data_list = data_list
        with open(self.data_list, 'r') as f:
            self.image_list = [x.strip() for x in f.readlines()]

        self.image_num = len(self.image_list)

    def __getitem__(self, item):
        image, im_size = load_image_test(os.path.join(self.data_root, self.image_list[item]))
        image = torch.Tensor(image)

        return {'image': image, 'name': self.image_list[item % self.image_num], 'size': im_size}

    def __len__(self):
        return self.image_num


def get_loader(config, mode='train', pin=False):
    shuffle = False
    if mode == 'train':
        shuffle = False
        dataset = ImageDataTrain(config.train_root, config.train_list)
        data_loader = data.DataLoader(dataset=dataset, batch_size=config.batch_size, shuffle=shuffle,
                                      num_workers=config.num_thread, pin_memory=pin)
    else:
        dataset = ImageDataTest(config.test_root, config.test_list)
        data_loader = data.DataLoader(dataset=dataset, batch_size=config.batch_size, shuffle=shuffle,
                                      num_workers=config.num_thread, pin_memory=pin)
    return data_loader


def load_image(path):
    if not os.path.exists(path):
        print('File {} not exists'.format(path))
    im = cv2.imread(path)
    in_ = np.array(im, dtype=np.float32)

    in_ -= np.array((104.00699, 116.66877, 122.67892))
    in_ = in_.transpose((2, 0, 1))  # CHW → HWC

    return in_


def load_image_test(path):
    if not os.path.exists(path):
        print('File {} not exists'.format(path))
    im = cv2.imread(path)
    in_ = np.array(im, dtype=np.float32)
    im_size = tuple(in_.shape[:2])
    in_ -= np.array((104.00699, 116.66877, 122.67892))
    in_ = in_.transpose((2, 0, 1))  # CHW → HWC
    return in_, im_size


def load_sal_label(path):
    if not os.path.exists(path):
        print('File {} not exists'.format(path))
    im = Image.open(path)
    label = np.array(im, dtype=np.float32)
    if len(label.shape) == 3:
        label = label[:, :, 0]
    label = label / 255.
    label = label[np.newaxis, ...]  # 添加通道维度
    return label


# 以 50% 的概率随机水平翻转图像
def cv_random_flip(img, label, edge):
    flip_flag = random.randint(0, 1)
    if flip_flag == 1:
        img = img[:, :, ::-1].copy()
        label = label[:, :, ::-1].copy()
        edge = edge[:, :, ::-1].copy()
    return img, label, edge


# 以 50% 的概率随机缩放图像
def generate_scale_label(image, label, edge):
    flip_flag = random.randint(0, 1)

    if flip_flag == 1:
        f_scale = 0.5 + random.randint(0, 11) / 10.0
        image = cv2.resize(image, None, fx=f_scale, fy=f_scale, interpolation=cv2.INTER_LINEAR)  # 双线性插值
        label = cv2.resize(label, None, fx=f_scale, fy=f_scale, interpolation=cv2.INTER_NEAREST)  # 最近邻插值
        edge = cv2.resize(edge, None, fx=f_scale, fy=f_scale, interpolation=cv2.INTER_NEAREST)  # 最近邻插值
        h, w, c = image.shape
        # image = np.reshape(image, (h, w, 3))
        # 保证图像的通道数为 1
        label = np.reshape(label, (h, w, 1))
        edge = np.reshape(edge, (h, w, 1))
    return image, label, edge


# 以 50% 的概率随机旋转图像
def random_rotate(x, y, z):
    flip_flag = random.randint(0, 1)
    if flip_flag == 1:
        angle = np.random.randint(-25, 25)
        # print('x',x.shape)
        h, w, c = x.shape
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)  # 旋转矩阵
        # cv2.warpAffine 进行仿射变换
        x = cv2.warpAffine(x, M, (w, h))
        y = cv2.warpAffine(y, M, (w, h))
        z = cv2.warpAffine(z, M, (w, h))
        y = np.reshape(y, (h, w, 1))
        z = np.reshape(z, (h, w, 1))
    return x, y, z


# 随机裁剪图像
def random_crop(x, y, z):
    h, w = y.shape
    randh = np.random.randint(h / 8)
    randw = np.random.randint(w / 8)
    # randf = np.random.randint(10)
    offseth = 0 if randh == 0 else np.random.randint(randh)
    offsetw = 0 if randw == 0 else np.random.randint(randw)
    p0, p1, p2, p3 = offseth, h + offseth - randh, offsetw, w + offsetw - randw
    return x[p0:p1, p2:p3], y[p0:p1, p2:p3], z[p0:p1, p2:p3]
