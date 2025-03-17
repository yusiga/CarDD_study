from torchvision import datasets, transforms
from torch.utils.data import Dataset
from matplotlib import pyplot as plt
import cv2
import numpy as np
import random
import scipy.ndimage as ndi
from tqdm import tqdm
import os
from PIL import Image


# 自定义数据集
class ScaledMNIST(Dataset):
    def __init__(self, train=True, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # 标记是训练集还是测试集

        if not os.path.exists('input/scaled_mnist_train.npz'):
            train_dataset = datasets.MNIST('~/data', train=True, download=True)
            train_imgs, train_labels = train_dataset.train_data.numpy(), train_dataset.train_labels.numpy()

            # 随机缩放
            scaled_train_imgs = []
            for i in tqdm(range(len(train_imgs))):
                img = np.pad(train_imgs[i], 14, 'constant')  # 在四周填充 14 个像素
                # 随机缩放，np.newaxis 使图像成为 (H, W, 1) 的形状，以便 random_zoom 处理
                img = random_zoom(img[:, :, np.newaxis], (0.5, 1.5))
                scaled_train_imgs.append(img[:, :, 0])  # 转换回 (H, W) 形状
            # 将 Python 列表 scaled_train_imgs 转换为 NumPy 数组，以便进行高效的数值计算和存储
            scaled_train_imgs = np.array(scaled_train_imgs)

            # 保存处理后的训练图片和标签，以便下次直接加载
            np.savez('input/scaled_mnist_train.npz', images=scaled_train_imgs, labels=train_labels)

        if not os.path.exists('input/scaled_mnist_test.npz'):
            test_dataset = datasets.MNIST('~/data', train=False, download=True)
            test_imgs, test_labels = test_dataset.test_data.numpy(), test_dataset.test_labels.numpy()

            scaled_test_imgs = []
            for i in tqdm(range(len(test_imgs))):
                img = np.pad(test_imgs[i], 14, 'constant')
                img = random_zoom(img[:, :, np.newaxis], (0.5, 1.5))
                scaled_test_imgs.append(img[:, :, 0])
            scaled_test_imgs = np.array(scaled_test_imgs)

            np.savez('input/scaled_mnist_test.npz', images=scaled_test_imgs, labels=test_labels)

        if self.train:
            scaled_mnist_train = np.load('input/scaled_mnist_train.npz')
            self.train_data = scaled_mnist_train['images']
            self.train_labels = scaled_mnist_train['labels']
        else:
            scaled_mnist_test = np.load('input/scaled_mnist_test.npz')
            self.test_data = scaled_mnist_test['images']
            self.test_labels = scaled_mnist_test['labels']

    def __getitem__(self, index):
        """
        获取单个样本
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # 将 NumPy 数组转换为 PIL 灰度图
        # 从而与返回 PIL 图像的其他数据集一致
        img = Image.fromarray(img, mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    # 返回数据集中样本的数量
    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)


# 将变换矩阵（如缩放、旋转等）调整到图像中心，确保变换操作相对于图像中心进行，而不是相对于左上角。
def transform_matrix_offset_center(matrix, x, y):
    # 计算图像中心的 x y 坐标
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    # 偏移矩阵 (offset_matrix)：将坐标原点从 (0,0) 平移到图像中心 (o_x, o_y)。
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    # 复位矩阵 (reset_matrix)：在变换完成后，将坐标原点移回 (0,0)，避免影响后续操作。
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix


def apply_transform(x, transform_matrix, channel_axis=0,
                    fill_mode='nearest', cval=0.):
    """
    将变换矩阵应用到图像，进行 仿射变换（如缩放、旋转等）。
    Args:
        x: 输入的图像（NumPy 数组）。
        transform_matrix: 变换矩阵（如缩放、旋转等）。
        channel_axis: 图像通道的索引位置（默认为 0）。
        fill_mode: 边界填充模式（用最近的像素填充）。
        cval: 填充值（默认填充 0，即黑色）。

    Returns:
        x: 变换后的图像
    """
    # 调整通道顺序，确保通道维度移至第 0 维前（即变成第 0 维），方便后续计算。(H, W, C) → (C, H, W)
    x = np.rollaxis(x, channel_axis, 0)
    # 由于变换矩阵是 3×3 的，我们只需要前 2×2 矩阵（用于旋转/缩放），以及最后一列的前两行（用于平移）。
    final_affine_matrix = transform_matrix[:2, :2]  # 提取 2x2 变换矩阵
    final_offset = transform_matrix[:2, 2]  # 提取平移向量
    # 对每个通道独立应用仿射变换。
    # ndi.interpolation.affine_transform 是 SciPy 库中的函数，可以进行仿射变换。
    # 其中 order=0 表示使用 最近邻插值，可以避免插值带来的模糊。
    # channel_images = [变换后的第 1 通道, 变换后的第 2 通道, ...]
    channel_images = [ndi.interpolation.affine_transform(
        x_channel,
        final_affine_matrix,
        final_offset,
        order=0,
        mode=fill_mode,
        cval=cval) for x_channel in x]
    # 恢复原始通道顺序
    # np.stack(..., axis=0) 的作用是把 channel_images 合并成新的数组，并让通道维度保持在第 0 维。
    # 通道维度移至第 channel_axis + 1 维前
    # x: (C, H, W) → (H, W, C)
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_axis + 1)
    return x


def random_zoom(X, zoom_range, row_axis=0, col_axis=1, channel_axis=2,
                fill_mode='nearest', cval=0.):
    """
    对输入图像进行随机缩放，缩放比例在 zoom_range 指定的范围内。
    Args:
        X: 输入的图像（NumPy 数组）。
        zoom_range: 缩放范围，例如 (0.5, 1.5) 表示 0.5x 到 1.5x 之间的随机缩放。
        row_axis
        col_axis: 行/列的索引（默认 0 和 1）。
        channel_axis: 图像通道的索引位置（默认为 2）。
        fill_mode: 边界填充模式（用最近的像素填充）。
        cval: 填充值（默认填充 0，即黑色）。

    Returns:
        X: 变换后的图像
    """
    # 检查 zoom_range 是否有效，必须是一个包含 两个浮点数 的元组。
    if len(zoom_range) != 2:
        raise ValueError('`zoom_range` should be a tuple or list of two floats. '
                         'Received arg: ', zoom_range)

    # 随机生成缩放因子 z，范围在 zoom_range 之间。
    z = np.random.uniform(zoom_range[0], zoom_range[1])

    # 构造缩放矩阵：
    # [[z, 0, 0], [0, z, 0], [0, 0, 1]]
    # 这表示 等比例缩放（x 轴和 y 轴都缩放 z 倍）。
    zoom_matrix = np.array([[z, 0, 0],
                            [0, z, 0],
                            [0, 0, 1]])

    # 获取 图像宽高，然后 调整变换矩阵，使其相对于中心缩放。
    h, w = X.shape[row_axis], X.shape[col_axis]
    transform_matrix = transform_matrix_offset_center(zoom_matrix, h, w)
    X = apply_transform(X, transform_matrix, channel_axis, fill_mode, cval)

    return X
