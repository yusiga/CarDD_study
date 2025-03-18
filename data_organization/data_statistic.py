import os
import matplotlib.pyplot as plt
from collections import defaultdict

# 设置全局字体（SimHei 为黑体，适用于 Windows）
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 或者 "Microsoft YaHei"
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题


# 统计每个类别（子文件夹）下的图片数量
def count_images_in_folders(root_dir):
    stats = {}  # 用于存储统计结果
    for subdir in os.listdir(root_dir):
        subdir_path = os.path.join(root_dir, subdir)
        if os.path.isdir(subdir_path):  # 确保是文件夹
            num_files = len([f for f in os.listdir(subdir_path) if os.path.isfile(os.path.join(subdir_path, f))])
            stats[subdir] = num_files  # 记录类别及对应的图片数
    return stats


# 统计 train、val、test
for split in ["train", "val", "test"]:
    split_dir = os.path.join("sorted_data", split)
    if os.path.exists(split_dir):
        stats = count_images_in_folders(split_dir)
        print(f"📊 {split} 数据集统计：")
        for category, count in stats.items():
            print(f"  - {category}: {count} 张图片")
        print("-" * 40)


# 统计整个数据集的文件总数
def count_total_files(directory):
    total = 0
    for root, dirs, files in os.walk(directory):
        total += len([f for f in files if os.path.isfile(os.path.join(root, f))])
    return total


dataset_dir = "sorted_data"
total_files = count_total_files(dataset_dir)
print(f"📌 数据集总文件数: {total_files} 张图片")


# 统计不同格式的文件数量
def count_files_by_extension(directory):
    ext_count = defaultdict(int)
    for root, dirs, files in os.walk(directory):
        for file in files:
            ext = os.path.splitext(file)[1].lower()  # 获取扩展名并转换为小写
            ext_count[ext] += 1
    return ext_count


file_stats = count_files_by_extension("sorted_data")
print("📌 各文件类型数量:")
for ext, count in file_stats.items():
    print(f"  - {ext}: {count} 个")

# 统计类别占比并可视化
# 统计 train 目录下的类别
train_stats = count_images_in_folders("sorted_data/train")

# 绘制饼图
labels = list(train_stats.keys())
sizes = list(train_stats.values())

plt.figure(figsize=(8, 6))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=['#ff9999', '#66b3ff', '#99ff99'])
plt.title("Train 数据集类别分布")
plt.axis("equal")  # 保持饼图是圆形
plt.show()