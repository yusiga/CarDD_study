import torch
from torch import nn
import torch.nn.functional as F
from networks.deeplab_resnet import \
    resnet50_locate
from networks.vgg import vgg16_locate

# convert: 负责通道转换（输入通道 → 输出通道）。
# deep_pool: 配置了深度池化层（DeepPoolLayer），控制池化通道数以及是否需要融合（need_fuse）。
# score: 评分层的通道数，用于预测输出。
config_vgg = {'convert': [[128, 256, 512, 512, 512], [64, 128, 256, 512, 512]],
              'deep_pool': [[512, 512, 256, 128], [512, 256, 128, 128], [True, True, True, False],
                            [True, True, True, False]],
              'score': 128}  # no convert layer, no conv6

config_resnet = {'convert': [[64, 256, 512, 1024, 2048], [128, 256, 256, 512, 512]],
                 'deep_pool': [[512, 512, 256, 256, 128], [512, 256, 256, 128, 128], [False, True, True, True, False],
                               [True, True, True, True, False]],
                 'score': 128}


# 作用: 通过 1x1 卷积调整特征图的通道数，便于后续 KRN 网络对输出特征图的操作。
# 输入: 多尺度（通道）特征图列表 list_x。
# 输出: 经过通道转换后的新特征图列表。
class ConvertLayer(nn.Module):
    def __init__(self, list_k):
        super(ConvertLayer, self).__init__()
        up = []
        for i in range(len(list_k[0])):
            up.append(nn.Sequential(nn.Conv2d(list_k[0][i], list_k[1][i], 1, 1, bias=False), nn.ReLU(inplace=True)))
        self.convert0 = nn.ModuleList(up)

    def forward(self, list_x):
        resl = []
        for i in range(len(list_x)):
            resl.append(self.convert0[i](list_x[i]))
        return resl

# SA Module 和 + ，进行特征融合
# 作用: 进行多尺度池化，并使用 bilinear 上采样恢复原尺寸。
# 输入: x: 当前特征图；x2: 更低一层的特征图（可选）。
# 输出: 经过池化和融合、bilinear 插值调整后的特征图。
class DeepPoolLayer_first(nn.Module):
    # (config['deep_pool'][0][i], config['deep_pool'][1][i], config['deep_pool'][2][i], config['deep_pool'][3][i])
    def __init__(self, k, k_out, need_x2,
                 need_fuse):
        super(DeepPoolLayer_first, self).__init__()
        self.pools_sizes = [2, 2, 2]
        self.need_x2 = need_x2
        self.need_fuse = need_fuse
        pools, convs = [], []
        # 平均池化 → 卷积层
        for i in self.pools_sizes:
            pools.append(nn.AvgPool2d(kernel_size=i, stride=i))
            convs.append(nn.Conv2d(k, k, 3, 1, 1, bias=False))
        self.pools = nn.ModuleList(pools)
        self.convs = nn.ModuleList(convs)
        self.relu = nn.ReLU()
        self.conv_sum = nn.Conv2d(k, k_out, 3, 1, 1, bias=False)
        if self.need_fuse:
            self.conv_sum_c = nn.Conv2d(k_out, k_out, 3, 1, 1, bias=False)

    def forward(self, x, x2=None):
        x_size = x.size()
        resl = x
        y = x
        for i in range(len(self.pools_sizes)):
            y = self.convs[i](self.pools[i](y))
            resl = torch.add(resl, F.interpolate(y, x_size[2:], mode='bilinear', align_corners=True))
        resl = self.relu(resl)
        if self.need_x2:
            resl = F.interpolate(resl, x2.size()[2:], mode='bilinear', align_corners=True)
        resl = self.conv_sum(resl)
        if self.need_fuse:
            resl = self.conv_sum_c(torch.add(resl, x2))
        return resl


# U
# 作用: 通过 1x1 卷积层生成边缘结果。
# 输入: x: 需要评分的特征图；x_size: 目标尺寸（可选）。
# 输出: 经过 bilinear 插值调整后的输出图。
class ScoreLayer(nn.Module):
    def __init__(self, k):
        super(ScoreLayer, self).__init__()
        self.score = nn.Conv2d(k, 1, 1, 1)

    def forward(self, x, x_size=None):
        x = self.score(x)  # shape: (batch, 1, H, W)
        # F.interpolate 是 PyTorch 的插值函数，作用是对 x 进行上采样或下采样。
        # x_size[2:] 获取目标的 height 和 width（x_size 形状为 (batch, channels, height, width)）。
        # align_corners=True：设为 True 时，插值后的像素与原图的角点对齐，避免缩放时的失真。
        # 用于调整特征图大小。
        if x_size is not None:
            x = F.interpolate(x, x_size[2:], mode='bilinear', align_corners=True)
        return x


# 选择 ResNet 或 VGG 作为主干网络，并返回相应的 ConvertLayer 和 ScoreLayer。
def extra_layer(base_model_cfg, vgg):
    if base_model_cfg == 'vgg':
        config = config_vgg
    elif base_model_cfg == 'resnet':
        config = config_resnet
    convert_layers, score_layers = [], []
    convert_layers = ConvertLayer(config['convert'])
    score_layers = ScoreLayer(config['score'])

    return vgg, convert_layers, score_layers


class KRN(nn.Module):
    def __init__(self, base_model_cfg, base, convert_layers, score_layers):
        super(KRN, self).__init__()
        self.base_model_cfg = base_model_cfg
        self.base = base  # 基本网络是一样的

        # self.deep_pool = nn.ModuleList(deep_pool_layers)
        self.score = score_layers
        if self.base_model_cfg == 'resnet':
            self.convert = convert_layers

        # SA 'deep_pool': [[512, 512, 256, 256, 128], [512, 256, 256, 128, 128],
        #                   [False, True, True, True, False], [True, True, True, True, False]]
        self.DeepPool_solid1 = DeepPoolLayer_first(512, 512, False, True)
        self.DeepPool_solid2 = DeepPoolLayer_first(512, 256, True, True)
        self.DeepPool_solid3 = DeepPoolLayer_first(256, 256, True, True)
        self.DeepPool_solid4 = DeepPoolLayer_first(256, 128, True, True)
        self.DeepPool_solid5 = DeepPoolLayer_first(128, 128, False, False)

        self.DeepPool_contour1 = DeepPoolLayer_first(512, 512, False, True)
        self.DeepPool_contour2 = DeepPoolLayer_first(512, 256, True, True)
        self.DeepPool_contour3 = DeepPoolLayer_first(256, 256, True, True)
        self.DeepPool_contour4 = DeepPoolLayer_first(256, 128, True, True)
        self.DeepPool_contour5 = DeepPoolLayer_first(128, 128, False, False)

        self.relu = nn.ReLU()

        # 调整图像通道数至 128 便于后续 KR Module 的处理
        self.conv_reduce1 = nn.Conv2d(512, 128, 1, 1, 1, bias=False)
        self.conv_reduce2 = nn.Conv2d(256, 128, 1, 1, 1, bias=False)
        self.conv_reduce3 = nn.Conv2d(256, 128, 1, 1, 1, bias=False)

        # 显著性图评估
        self.score_solid = ScoreLayer(128)
        self.score_contour = ScoreLayer(128)
        self.score_solid1 = ScoreLayer(512)
        self.score_contour1 = ScoreLayer(512)
        self.score_solid2 = ScoreLayer(256)
        self.score_contour2 = ScoreLayer(256)
        self.score_solid3 = ScoreLayer(256)
        self.score_contour3 = ScoreLayer(256)
        self.score_solid4 = ScoreLayer(128)
        self.score_contour4 = ScoreLayer(128)
        self.score_sum_out = ScoreLayer(128)

        self.conv_1 = nn.Conv2d(512, 512, 3, 1, 1, bias=False)
        self.conv_2 = nn.Conv2d(256, 256, 3, 1, 1, bias=False)
        self.conv_3 = nn.Conv2d(256, 256, 3, 1, 1, bias=False)
        self.conv_4 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)

        self.conv_add1 = nn.Conv2d(128, 32, 3, 1, 1, bias=False)
        self.conv_add2 = nn.Conv2d(128, 32, 3, 1, 1, bias=False)
        self.conv_add3 = nn.Conv2d(128, 32, 3, 1, 1, bias=False)
        self.conv_add4 = nn.Conv2d(128, 32, 3, 1, 1, bias=False)

        self.conv_sum_out = nn.Conv2d(128, 128, 3, 1, 1, bias=False)

    def forward(self, x):
        x_size = x.size()  # (B, 3, H, W)
        conv2merge, infos = self.base(x)  # 提取 ResNet 预训练模型的特征。
        if self.base_model_cfg == 'resnet':
            conv2merge = self.convert(conv2merge)
        # 反转特征层，使高层特征先处理。
        # 通道数：2048 1024 512 256 64，高度和宽度：/32 /16 /8 /4 /2
        conv2merge = conv2merge[::-1]

        # 自高层至底层逐层的特征融合 → 显著性图评估 → 调整图像通道数至 128 便于后续 KR Module 的处理
        # (B, 512, H/32, W/32)
        merge_solid1 = self.DeepPool_solid1(conv2merge[0], conv2merge[1])
        out_merge_solid1 = self.score_solid1(merge_solid1, x_size)
        out_merge_solid1 = F.sigmoid(out_merge_solid1)
        fea_reduce1 = self.conv_reduce1(merge_solid1)
        fea_reduce1 = self.relu(fea_reduce1)

        # (B, 256, H/16, W/16)
        merge_solid2 = self.DeepPool_solid2(merge_solid1, conv2merge[2])
        out_merge_solid2 = self.score_solid2(merge_solid2, x_size)
        out_merge_solid2 = F.sigmoid(out_merge_solid2)
        fea_reduce2 = self.conv_reduce2(merge_solid2)
        fea_reduce2 = self.relu(fea_reduce2)

        # (B, 256, H/8, W/8)
        merge_solid3 = self.DeepPool_solid3(merge_solid2, conv2merge[3])
        out_merge_solid3 = self.score_solid3(merge_solid3, x_size)
        out_merge_solid3 = F.sigmoid(out_merge_solid3)
        fea_reduce3 = self.conv_reduce3(merge_solid3)
        fea_reduce3 = self.relu(fea_reduce3)

        # (B, 128, H/4, W/4)
        merge_solid4 = self.DeepPool_solid4(merge_solid3, conv2merge[4])
        out_merge_solid4 = self.score_solid4(merge_solid4, x_size)
        out_merge_solid4 = F.sigmoid(out_merge_solid4)
        fea_reduce4 = merge_solid4

        # 融合全部的特征至 merge_solid5
        # (B, 128, H/2, W/2)
        merge_solid5 = self.DeepPool_solid5(merge_solid4)
        merge_solid = self.score_solid(merge_solid5, x_size)
        merge_solid = F.sigmoid(merge_solid)

        # 完成 U
        # F.interpolate 是 PyTorch 的插值函数，作用是对 fea_reduce 进行上采样或下采样。
        # [2:] 获取目标的 height 和 width 。
        # align_corners=True：设为 True 时，插值后的像素与原图的角点对齐，避免缩放时的失真。
        # 用于调整特征图大小。
        fea_reduce1 = F.interpolate(fea_reduce1, merge_solid5.size()[2:], mode='bilinear', align_corners=True)
        fea_reduce2 = F.interpolate(fea_reduce2, merge_solid5.size()[2:], mode='bilinear', align_corners=True)
        fea_reduce3 = F.interpolate(fea_reduce3, merge_solid5.size()[2:], mode='bilinear', align_corners=True)
        fea_reduce4 = F.interpolate(fea_reduce4, merge_solid5.size()[2:], mode='bilinear', align_corners=True)
        # 下面两段代码完成 KR Module 中的 A
        # torch.add 即 Pixel-wise add，用于特征融合
        fea_add1 = torch.add(merge_solid5, fea_reduce1)
        fea_add2 = torch.add(merge_solid5, fea_reduce2)
        fea_add3 = torch.add(merge_solid5, fea_reduce3)
        fea_add4 = torch.add(merge_solid5, fea_reduce4)
        # conv_add 即 3x3 卷积核
        fea_add1 = self.conv_add1(fea_add1)
        fea_add1 = self.relu(fea_add1)
        fea_add2 = self.conv_add2(fea_add2)
        fea_add2 = self.relu(fea_add2)
        fea_add3 = self.conv_add3(fea_add3)
        fea_add3 = self.relu(fea_add3)
        fea_add4 = self.conv_add4(fea_add4)
        fea_add4 = self.relu(fea_add4)
        feasum_out = torch.cat((fea_add1, fea_add2, fea_add3, fea_add4), 1)
        feasum_out = self.conv_sum_out(feasum_out)
        feasum_out = self.score_sum_out(feasum_out, x_size)
        feasum_out = F.sigmoid(feasum_out)  # 归一化，更适合二分类分割任务如 SOD

        return feasum_out, merge_solid, out_merge_solid1, out_merge_solid2, out_merge_solid3, out_merge_solid4


def build_model(base_model_cfg='vgg'):
    if base_model_cfg == 'vgg':
        return KRN(base_model_cfg, *extra_layer(base_model_cfg, vgg16_locate()))
    elif base_model_cfg == 'resnet':
        return KRN(base_model_cfg, *extra_layer(base_model_cfg, resnet50_locate()))


# 初始化 Conv2d 权重，使其符合高斯分布 N(0, 0.01)，并将 bias 置零。
def weights_init(m):
    if isinstance(m, nn.Conv2d):
        m.weight.data.normal_(0, 0.01)
        if m.bias is not None:
            m.bias.data.zero_()
