***CarDD: A New Dataset for Vision-Based Car Damage Detection***

# Motivation

自动汽车损伤评估的目标是对车辆上的损伤进行定位和分类，并通过绘制损伤的精确位置将其可视化，本质上是损伤的目标检测和实例分割。目前存在问题。

# Method

## 数据集

**流程：**

![数据集构建流程](D:\PycharmProjects\CarDD_study\image\数据集构建流程.png)

### 1. 图像采集和选择

- 删除重复图像：Duplicate Cleaner

- 图像中是否包含损坏的汽车：VGG-16二分类 + 人工

### 2. 图像标注

标注准则：

- 损伤类别间的优先级：针对混合损伤，裂缝＞凹痕＞划痕（在前面的损伤会更难修复
- 不同部件上的相同损伤类别被标注为多个实例
- 同一部件上的相同损伤被合并为一个实例

### 3. 统计数据

 分析数据集特征。

- 4000张受损汽车图像，超过9000个受标记的损伤实例
- 对象大小：128×128 256×256 → 小 / 中 / 大
- 图像分辨率：高分辨率的图像更容易发现全部的损坏实例

# Experiment

## 实例分割和目标检测实验

### 模型-DCN+

**模型流程：**

![DCN+流程](D:\PycharmProjects\CarDD_study\image\DCN+流程.png)

#### 1. 关键技术

- **多尺度学习（multi-scale learning）** —— 处理目标的尺度多样性。
- **焦点损失（focal loss）** —— 强化模型对难分类别的关注度。

#### 2. 预测结果

- 目标类别（class）
- 目标的边界框位置（bounding box location）
- 目标的掩码（mask）

#### 3. 模型优化

损失计算基于预测结果和真实标注进行，通过最小化焦点损失、L1损失、交叉熵损失之和来优化模型

#### 4. 焦点损失

采用 α 平衡版本（α-balanced version）来控制**不同类别的重要性**，具体公式如下：

$$
L_{focal} = -\alpha(1 - p_{i,c})^\gamma \log(p_{i,c})
$$

pi,c 表示目标 i 在其真实类别上的预测概率。

### 实验设置

#### 1. 数据集

训练集 2816（70.4%）+ 验证集 810（20.25%）+ 测试集 374（9.35%） - 7：2：1

#### 2. 评估指标

IoU 阈值从 0.5-0.95 ，以 0.05 为间隔计算多个 AP 的均值。

- 实例分割：掩码 AP
- 目标检测：边界框 AP

#### 3. 训练细节

- NVIDIA Tesla P100
- batch_size 8
- epoches 24 （1-16：0.01 / 17-22：0.001 / 23-24：0.0001）
- weight decay 0.0001
- momentum 0.9（SGD + momentum）

### 实验参数

图像高度 = [640, 1200] / 宽度 = 1333

α = 0.50 / γ = 2.0

### 消融实验

验证多尺度学习和焦点损失的有效性。

多尺度学习：提高了对低质量输入的损伤检测效果

焦点损失：提升APs，得益于凹痕、划痕、裂缝的提升

## 显著目标检测实验

使用SOD方法提升对凹痕、划痕、裂缝的定位和边界确定。

### 评价指标

#### 1. F-measure

$$
F_{\beta} = \frac{(1 + \beta^2) \cdot \text{Precision} \cdot \text{Recall}}{\beta^2 \cdot \text{Precision} + \text{Recall}}
$$

- **Precision（查准率）**：预测为目标的像素中，实际正确的比例。

- **Recall（召回率）**：实际目标像素中，被正确预测的比例。

- **β** 控制 Precision 和 Recall 的权重，通常设置为 0.3（更关注 Precision）。

衡量模型在**精准度和召回率**之间的平衡。

#### 2. Weighted F-measure

$$
F_wβ = \frac{(1 + \beta^2) \cdot \text{Precision}_w \cdot \text{Recall}_w}{\beta^2 \cdot \text{Precision}_w + \text{Recall}_w}
$$

在显著区域（Salient Region）和背景区域（Non-Salient Region）之间分配不同的权重，以降低背景噪声的影响。

#### 3. S-measure

$$
S_m = \alpha \cdot S_o + (1 - \alpha) \cdot S_r
$$

- **S_o**：目标结构相似性（Object-aware similarity）。

- **S_r**：区域结构相似性（Region-aware similarity）。

- **α** 是一个权重参数，通常设为 0.5。

衡量预测结果与真实目标在**全局结构**上的匹配程度，比单纯的像素级比较更符合人眼感知。

#### 4. E-measure

$$
E_m = \frac{1}{W \times H} \sum_{x=1}^{W} \sum_{y=1}^{H} \Phi (S(x,y), G(x,y))
$$

- **S(x, y)** 是预测的显著性值。

- **G(x, y)** 是真实的显著性标签。

- **Φ()** 是非线性转换函数，使得高亮区域和真实目标对齐得更好。

用于衡量整体显著性预测与真实目标的**对齐程度**，相比 F-measure，E-measure 更注重显著性区域的整体感知，能够同时捕捉全局信息和局部细节。

#### 5. Mean Absolute Error（MAE）

$$
MAE = \frac{1}{W \times H} \sum_{x=1}^{W} \sum_{y=1}^{H} |S(x,y) - G(x,y)|
$$

- **W, H** 是图像的宽高。

- **S(x,y)** 是预测的显著性值。

- **G(x,y)** 是真实的显著性标签

衡量预测显著性图（Saliency Map）和真实标签（Ground Truth）之间的**像素级**误差。

# Conclusion

## 主要贡献

- 提出了第一个公开的新的汽车损伤检测数据集，用于基于视觉的汽车损伤检测和分割。数据集包含高质量的图像，其中标注了**损坏类型、损坏位置和损坏程度**。
- 在数据集上做了广泛的实验，采用最先进的深度方法进行不同的任务（分类、目标检测、实例分割、显著目标检测）。基于SOTA方法不理想的表现，提出了**DCN+**，显著提高了目标检测和实例分割的性能。
- 首次利用SOD方法来处理汽车损伤检测。

## 六种汽车损坏类别

- dent 凹痕
- scratch 划痕
- crack 裂缝
- glass shatter 玻璃破碎
- lamp broken 车灯损坏
- tire flat 轮胎漏气

## 未来工作

对于前三种汽车损坏类别，目前模型进行目标检测和实例分割的效果欠佳。可能的原因：这类实例的对象尺度变化多端且普遍比较小；凹痕、划痕、裂缝经常糅合在一起，且不容易相互区分。即物体规模和形状的多样性、小尺寸物体、灵活的边界。

# 一些定义

## 1. Object Detection（目标检测）

**定义**：目标检测是一种计算机视觉技术，旨在识别图像或视频中的目标对象，并为每个目标生成**边界框（bounding box）**及其类别标签。

**特点**：

- 识别目标的类别
- 确定目标的位置（使用边界框）
- 适用于物体数量不固定的场景

**范例**： 假设一张街道图片中有**两辆车**和**一个行人**，目标检测算法会输出：

- 车（Bounding Box: (x1, y1, x2, y2)）
- 车（Bounding Box: (x3, y3, x4, y4)）
- 行人（Bounding Box: (x5, y5, x6, y6)）

**典型算法**：

- YOLO（You Only Look Once）
- Faster R-CNN
- SSD（Single Shot MultiBox Detector）

------

## 2. Instance Segmentation（实例分割）

**定义**：实例分割是在目标检测的基础上，进一步将图像中的目标区域**逐像素分割**，即每个目标的轮廓边界都会被精确划分出来，而不是简单的矩形框。

**特点**：

- 识别目标的类别
- 精确分割目标的轮廓（像素级）
- 适用于多个同类对象的检测，如检测多个人，并区分彼此

**范例**： 在上面的街道场景中，实例分割会：

- 生成两辆车的像素级区域（不会混淆）
- 生成行人的像素级区域
- 这样可以清楚地区分**不同实例**的物体，而不是简单的边界框

**典型算法**：

- Mask R-CNN
- YOLACT（You Only Look At Coefficients）
- SOLO（Segmenting Objects by Locations）

------

## 3. Salient Object Detection (SOD)（显著目标检测）

**定义**：显著目标检测的目标是找到图像中**最显著（最吸引注意力）**的对象，而不关注其类别或实例区分。通常用于背景模糊处理、图像编辑等任务。

**特点**：

- 关注**人类视觉注意力**，找出最引人注目的物体
- 不一定需要分类
- 主要用于图像前景/背景分离

**范例**： 给定一张图片，其中有一只鸟站在枝头，SOD 任务会：

- 生成一个**显著性图**（Saliency Map），其中鸟的区域亮度高，背景暗淡
- 这可以用于自动裁剪、焦点调整等

**典型算法**：

- U2-Net
- BASNet（Boundary-Aware Salient Object Detection）
- PoolNet（Pyramid Pooling Module）

***

## 4. mAP（Mean Average Precision，均值平均精度）

mAP 是目标检测和信息检索任务中的核心评估指标，用于衡量模型对多个类别的检测效果。它结合了精确率（Precision）和召回率（Recall），综合评估模型的准确性和稳定性。

### 计算步骤

**计算 Precision 和 Recall**

- Precision（精确率） = 预测正确的目标 / 被预测的所有目标（正检 + 误检）
- Recall（召回率） = 预测正确的目标 / 实际正确的所有目标（正检 + 漏检）

**计算 AP（Average Precision）**

AP 是基于 Precision-Recall 曲线 计算的。步骤：

1. 以不同的**置信度阈值**（Confidence Threshold）计算 Precision 和 Recall，绘制 PR 曲线。
2. 计算 PR 曲线下的面积，得到该类别的 AP（平均精度）。

置信度阈值决定保留哪些预测，影响 Precision 和 Recall：
- 阈值高（只保留高置信度的预测）→ Precision 高，Recall 低。
- 阈值低（保留更多预测）→ Recall 高，Precision 可能降低。

**计算 mAP（Mean Average Precision）**

对于多类别检测任务，计算所有类别的 AP 均值。

***

## 5. ground-truth binary maps

真实标注的二值图，用于表示数据集中某种损坏类别、位置或轮廓的标注结果。具体来说：

- **ground-truth**：指的是“真实标注”或“人工标注的标准答案”，通常由专家或高质量数据标注工具生成。
- **binary maps**：指的是“二值图”，即每个像素只有两种可能的值（例如 0 和 1）。在损坏检测任务中，这通常表示损坏区域（1）和非损坏区域（0）。

***

## 6. 消融实验

类似于控制变量。
