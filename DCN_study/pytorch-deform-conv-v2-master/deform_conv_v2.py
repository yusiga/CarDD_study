import torch
from torch import nn


class DeformConv2d(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1, bias=None, modulation=False):
        """
        Args:
            modulation (bool, optional): If True, Modulated Defomable Convolution (Deformable ConvNets v2).
        """
        super(DeformConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.zero_padding = nn.ZeroPad2d(padding)

        # 标准卷积层，用于最终处理变形后的特征图
        self.conv = nn.Conv2d(inc, outc, kernel_size=kernel_size, stride=kernel_size, bias=bias)

        # 学习偏移量的卷积层，最终输出：(batch, 2N, H/stride, W/stride)
        self.p_conv = nn.Conv2d(inc, 2 * kernel_size * kernel_size, kernel_size=3, padding=1, stride=stride)
        nn.init.constant_(self.p_conv.weight, 0)  # 初始化为0，即初始时为普通卷积，偏移量为0
        self.p_conv.register_full_backward_hook(self._set_lr)  # 注册反向传播的Hook，并在反向传播时执行_set_lr这个函数

        # 学习调制因子的卷积层，最终输出：(batch, k*k, H/stride, W/stride) N = k*k
        self.modulation = modulation
        if modulation:
            self.m_conv = nn.Conv2d(inc, kernel_size * kernel_size, kernel_size=3, padding=1, stride=stride)
            nn.init.constant_(self.m_conv.weight, 0)
            self.m_conv.register_full_backward_hook(self._set_lr)

    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))

    def forward(self, x):
        offset = self.p_conv(x)  # (batch, 2N, h/stride, w/stride)
        if self.modulation:
            m = torch.sigmoid(self.m_conv(x))  # 经过 sigmoid() 归一化，使得调制因子范围在 [0,1]

        dtype = offset.data.type()  # 获取 offset 的数据类型，后面用于创建与 dtype 相同的新张量，以保持计算一致性
        ks = self.kernel_size
        N = offset.size(1) // 2  # 卷积核中的采样点数量（卷积核元素个数）

        if self.padding:
            x = self.zero_padding(x)

        # 计算偏移后的采样点（卷积核中心点 + 偏移量）
        # (b, 2N, h, w)
        p = self._get_p(offset, dtype)

        # 双线性插值部分
        # contiguous() 确保张量在内存中是连续的，以便后续计算。permute(0, 2, 3, 1)交换维度 (b, 2N, h, w) → (b, h, w, 2N)
        p = p.contiguous().permute(0, 2, 3, 1)
        # detach() 切断计算图，防止 q_lt 影响 p 的梯度计算，减少梯度开销。
        # .floor() 向下取整，得到 p 左上角的整数网格坐标，表示双线性插值的 q_lt（左上角插值点）。
        q_lt = p.detach().floor()
        q_rb = q_lt + 1  # 直接在 q_lt 上加 1，得到右下角的整数网格点。

        # 计算偏移采样点周围四个点的坐标
        # q_lt[..., :N] 取 q_lt 的前 N 维，即 x 坐标
        # q_lt[..., N:] 取 q_lt 的后 N 维，即 y 坐标
        # torch.clamp(..., 0, x.size(2)-1) 限制 x 和 y 坐标在图像范围内
        # torch.cat(..., dim=-1) 重新拼接回 (B, H, W, 2N) 形状
        # .long() 确保索引是整数
        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2) - 1), torch.clamp(q_lt[..., N:], 0, x.size(3) - 1)],
                         dim=-1).long()
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2) - 1), torch.clamp(q_rb[..., N:], 0, x.size(3) - 1)],
                         dim=-1).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)

        # clip p
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2) - 1), torch.clamp(p[..., N:], 0, x.size(3) - 1)], dim=-1)

        # 根据偏移点 p 在四个最近网格点 q_lt, q_rb, q_lb, q_rt 之间的相对位置（即距离），计算四个点的贡献权重。
        # bilinear kernel (b, h, w, N)。
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

        # 获取采样点的像素值
        # (b, c, h, w, N)
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        # (b, c, h, w, N)
        # unsqueeze：在指定维度位置插入一个长度为1的维度，计算 p 位置的最终插值像素值 x_offset
        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt

        # modulation
        if self.modulation:
            m = m.contiguous().permute(0, 2, 3, 1)
            m = m.unsqueeze(dim=1)
            m = torch.cat([m for _ in range(x_offset.size(1))], dim=1)  # 沿着通道复制 m c次
            x_offset *= m  # 通过调制因子控制像素的权重

        # 整形，方便卷积计算 → (batch_size, in_channel, kernel*h, kernel*w)
        x_offset = self._reshape_x_offset(x_offset, ks)
        out = self.conv(x_offset)

        return out

    # 返回了一个卷积核内部的坐标网格，以卷积核中心p_0为原点。
    # 如果是 3×3 卷积核的话，横纵坐标刻度应该都是 [-1, 0, 1]，这是相对卷积核中心的相对坐标。
    def _get_p_n(self, N, dtype):
        p_n_x, p_n_y = torch.meshgrid(
            torch.arange(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1),
            torch.arange(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1),
            indexing = "ij"
        )
        # (2N, 1)
        p_n = torch.cat([torch.flatten(p_n_x), torch.flatten(p_n_y)], 0)
        p_n = p_n.view(1, 2 * N, 1, 1).type(dtype)

        return p_n

    # 返回了所有卷积核中心p_0在 input feature map 上的绝对坐标值。
    # p_0的个数刚好等于 output feature map 的元素个数。
    # 从而根据 stride 和输出大小，能够确定原始卷积核中心在 input feature map 上的坐标位置：卷积核中心之间相隔 stride，共 h*w个。
    def _get_p_0(self, h, w, N, dtype):
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(1, h * self.stride + 1, self.stride),
            torch.arange(1, w * self.stride + 1, self.stride),
            indexing = "ij"
        )
        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)  # 复制N份
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)

        return p_0

    # 获取最终的采样点
    def _get_p(self, offset, dtype):
        N, h, w = offset.size(1) // 2, offset.size(2), offset.size(3)  # N h w

        # (1, 2N, 1, 1)
        p_n = self._get_p_n(N, dtype)
        # (1, 2N, h, w)
        p_0 = self._get_p_0(h, w, N, dtype)
        p = p_0 + p_n + offset
        return p

    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)  # 原始特征图的宽度
        c = x.size(1)  # 原始特征图的通道数
        # 将x从 (b, c, h, w) → (b, c, h*w)
        x = x.contiguous().view(b, c, -1)

        # 计算偏移采样点的索引，将 2D 坐标 (x, y) 转换为 1D 索引
        # (b, h, w, N)
        index = q[..., :N] * padded_w + q[..., N:]  # offset_x*w + offset_y
        # (b, c, h*w*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)

        # dim=-1 表示沿着展平的 h*w 维度采样
        # index 里存放的是 x 中要采样的索引
        # 从 x 中取出 q 对应的偏移位置上的值
        # .view(b, c, h, w, N)：采样后的 x_offset 变回 (b, c, h, w, N)，每个 (h, w) 位置有 N 个偏移点对应的特征值
        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)

        return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset, ks):
        b, c, h, w, N = x_offset.size()
        x_offset = torch.cat([x_offset[..., s:s + ks].contiguous().view(b, c, h, w * ks) for s in range(0, N, ks)],
                             dim=-1)
        x_offset = x_offset.contiguous().view(b, c, h * ks, w * ks)

        return x_offset
