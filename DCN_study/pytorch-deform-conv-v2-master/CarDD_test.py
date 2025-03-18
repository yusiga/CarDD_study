import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

from utils import *
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
    parser.add_argument('--epochs', default=24, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--optimizer', default='SGD',
                        choices=['Adam', 'SGD'],
                        help='loss: ' +
                             ' | '.join(['Adam', 'SGD']) +
                             ' (default: Adam)')
    parser.add_argument('--lr', '--learning-rate', default=1e-2, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight-decay', default=1e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', default=False, type=str2bool,
                        help='nesterov')

    args = parser.parse_args()

    return args


# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

args = parse_args()

num_classes = 5
# 初始化模型
model = archs.__dict__[args.arch](args, num_classes)
# 加载权重
model.load_state_dict(torch.load("/content/drive/MyDrive/models/%s/model.pth" % args.name, map_location=device))  # 加载权重
model.to(device)
model.eval()

# 预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5105, 0.4894, 0.4883), (0.2846, 0.2799, 0.2816))
])

# 载入测试集
test_dataset = datasets.ImageFolder(root="/content/drive/MyDrive/data_set/sorted_data/test", transform=transform)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=2)

# 计算整体测试集准确率
correct = 0
total = 0
all_labels = []
all_preds = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())

accuracy = correct / total
print(f"测试集整体准确率: {accuracy:.4f}")

# 计算每个类别的准确率
class_correct = np.zeros(5)  # 5个类别
class_total = np.zeros(5)

for i in range(len(all_labels)):
    label = all_labels[i]
    class_correct[label] += (all_preds[i] == label)
    class_total[label] += 1

print("每个类别的准确率:")
for i in range(5):
    acc = class_correct[i] / class_total[i] if class_total[i] > 0 else 0
    print(f"类别 {i + 1}: {acc:.4f}")

# 计算混淆矩阵
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[1, 2, 3, 4, 5], yticklabels=[1, 2, 3, 4, 5])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

# 计算 Precision, Recall, F1-Score
print("分类报告:")
report = classification_report(all_labels, all_preds,
                               target_names=["Class 1", "Class 2", "Class 3", "Class 4", "Class 5"])
print(report)


# 预测单张图片
def predict_image(image_path):
    image = Image.open(image_path)
    plt.imshow(image)
    plt.axis("off")  # 关闭坐标轴
    plt.show()

    image = transform(image).unsqueeze(0).to(device)  # 预处理并增加 batch 维度

    model.eval()
    with torch.no_grad():
        output = model(image).squeeze(0)  # 直接去掉 batch 维度
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict, dim=0).item() + 1  # 先转为 Python int 避免 NumPy 相关问题
        predict_prob = predict[predict_cla - 1].detach().cpu().item()  # 确保数值在 CPU 并转换为 Python float

    print(f"预测类别: {predict_cla}, 预测概率: {predict_prob:.4f}")


# 测试单张图片预测（请替换路径）
predict_image("/content/drive/MyDrive/data_set/sorted_data/test/1/000012.jpg")
