import os
from PIL import Image
import torch
from torchvision import transforms
from tqdm import tqdm  # 进度条

# 你的数据集路径
data_dir = "sorted_data/train"

# 定义转换：只转 Tensor，不改变大小
transform = transforms.ToTensor()

# 初始化累积和
# mean_sum = torch.zeros(3)  # 返回一个 形状为 (3,) 的全零张量
# std_sum = torch.zeros(3)
# num_pixels = 0

pixel_sum = torch.zeros(3)
pixel_sq_sum = torch.zeros(3)
num_pixels = 0

# 遍历所有图片
for class_folder in os.listdir(data_dir):  # 遍历类别文件夹
    class_path = os.path.join(data_dir, class_folder)
    if not os.path.isdir(class_path):
        continue

    for img_name in tqdm(os.listdir(class_path), desc=f"Processing {class_folder}"):
        img_path = os.path.join(class_path, img_name)
        image = Image.open(img_path).convert("RGB")  # 确保是 RGB 格式
        tensor = transform(image)  # 转换为 Tensor，shape: [C, H, W]

#         # 计算每张图像的像素总和
#         mean_sum += tensor.mean(dim=[1, 2])  # 沿着宽度和高度维度计算每个通道的均值 (R, G, B)
#         std_sum += tensor.std(dim=[1, 2])  # 沿着宽度和高度维度计算每个通道的标准差 (R, G, B)
#         num_pixels += 1  # 统计图像数量
#
# # 计算最终均值和标准差
# mean = mean_sum / num_pixels
# std = std_sum / num_pixels

        pixel_sum += tensor.sum(dim=[1, 2])  # 累加像素值
        pixel_sq_sum += (tensor ** 2).sum(dim=[1, 2])  # 累加平方值
        num_pixels += tensor.shape[1] * tensor.shape[2]  # 计算总像素点数

mean = pixel_sum / num_pixels
std = torch.sqrt(pixel_sq_sum / num_pixels - mean ** 2)  # 计算标准差

print("数据集均值:", mean.tolist())  # [R, G, B]
print("数据集标准差:", std.tolist())  # [R, G, B]
