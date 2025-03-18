import os
import shutil
import pandas as pd

# 配置路径
xlsx_path = "CarDD_COCO/annotations/image_info.xlsx"  # xlsx 文件路径
base_dir = "CarDD_COCO"  # 数据集根目录
splits = ["train", "val", "test"]  # 训练集、验证集、测试集的文件夹名称
dest_root = "sorted_data"  # 目标存放路径

# 读取 Excel 文件
df = pd.read_excel(xlsx_path)

# 遍历 train、val、test 目录
for split in splits:
    src_dir = os.path.join(base_dir, split)  # 原始数据目录，如 dataset/train
    dest_dir = os.path.join(dest_root, split)  # 目标存放路径，如 sorted_dataset/train

    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)  # 如果目标根目录不存在，先创建

    # 遍历 Excel 记录，将文件按类别归类
    for _, row in df.iterrows():
        filename = row["file_name"]  # 图片文件名
        category = str(row["#categories"])  # 图片类别（转换为字符串，避免 int 类型错误）

        src_path = os.path.join(src_dir, filename)  # 原始路径
        category_dir = os.path.join(dest_dir, category)  # 目标类别文件夹路径
        dest_path = os.path.join(category_dir, filename)  # 目标路径

        # 检查原始图片是否存在
        if os.path.exists(src_path):
            if not os.path.exists(category_dir):
                os.makedirs(category_dir)  # 创建类别文件夹
            shutil.move(src_path, dest_path)  # 移动文件
            print(f"Moved {filename} -> {category_dir}")
        else:
            print(f"Warning: {filename} not found in {src_dir}")

print("All files sorted successfully!")