# -*- coding: utf-8 -*-
# @Time    : 2020
# @Author  : Lart Pang
# @GitHub  : https://github.com/lartpang
from numbers import Number

import torch
import torch.nn.functional as F


def cus_sample(
    feat: torch.Tensor,
    mode=None,
    factors=None,
    *,
    interpolation="bilinear",
    align_corners=False,
) -> torch.Tensor:
    """
    :param feat: 输入特征
    :param mode: size/scale 大小/比例
    :param factors: shape list for mode=size or scale list for mode=scale 目标大小/比例因子
    :param interpolation:
    :param align_corners: 具体差异可见https://www.yuque.com/lart/idh721/ugwn46 角点的对齐方式
    :return: the resized tensor
    """
    if mode is None:
        return feat
    else:
        if factors is None:
            raise ValueError(
                f"factors should be valid data when mode is not None, but it is {factors} now."
                f"feat.shape: {feat.shape}, mode: {mode}, interpolation: {interpolation}, align_corners: {align_corners}"
            )

    interp_cfg = {}
    if mode == "size":
        if isinstance(factors, Number):
            factors = (factors, factors)
        assert isinstance(factors, (list, tuple)) and len(factors) == 2
        factors = [int(x) for x in factors]
        if factors == list(feat.shape[2:]):
            return feat
        interp_cfg["size"] = factors
    elif mode == "scale":
        assert isinstance(factors, (int, float))
        if factors == 1:
            return feat
        recompute_scale_factor = None
        if isinstance(factors, float):
            recompute_scale_factor = False
        interp_cfg["scale_factor"] = factors
        interp_cfg["recompute_scale_factor"] = recompute_scale_factor
    else:
        raise NotImplementedError(f"mode can not be {mode}")

    if interpolation == "nearest":
        if align_corners is False:
            align_corners = None
        assert align_corners is None, (
            "align_corners option can only be set with the interpolating modes: "
            "linear | bilinear | bicubic | trilinear, so we will set it to None"
        )
    try:
        # 使用 F.interpolate 来执行实际的上采样操作
        # interp_cfg 中存储了调整大小的配置。
        result = F.interpolate(feat, mode=interpolation, align_corners=align_corners, **interp_cfg)
    except NotImplementedError as e:
        print(
            f"shape: {feat.shape}\n"
            f"mode={mode}\n"
            f"factors={factors}\n"
            f"interpolation={interpolation}\n"
            f"align_corners={align_corners}"
        )
        raise e
    except Exception as e:
        raise e
    return result


# 将所有输入的张量（除了最后一个）上采样到最后一个张量的大小，并将它们逐个相加
def upsample_add(*xs: torch.Tensor, interpolation="bilinear", align_corners=False) -> torch.Tensor:
    """
    resize xs[:-1] to the size of xs[-1] and add them together.

    Args:
        xs:
        interpolation: config for cus_sample
        align_corners: config for cus_sample
    """
    y = xs[-1]
    for x in xs[:-1]:
        y = y + cus_sample(
            x, mode="size", factors=y.size()[2:], interpolation=interpolation, align_corners=align_corners
        )
    return y


# 将所有输入的张量（除了最后一个）上采样到最后一个张量的大小，并将上采样后的张量拼接（cat）而不是相加。
def upsample_cat(*xs: torch.Tensor, interpolation="bilinear", align_corners=False) -> torch.Tensor:
    """
    resize xs[:-1] to the size of xs[-1] and concat them together.

    Args:
        xs:
        interpolation: config for cus_sample
        align_corners: config for cus_sample
    """
    y = xs[-1]
    out = []
    for x in xs[:-1]:
        out.append(
            cus_sample(x, mode="size", factors=y.size()[2:], interpolation=interpolation, align_corners=align_corners)
        )
    return torch.cat([*out, y], dim=1)


# 将输入 b 上采样到与 a 相同的大小，并将其通道数与 a 进行处理，最后返回它们的和
def upsample_reduce(b, a, interpolation="bilinear", align_corners=False) -> torch.Tensor:
    """
    上采样所有特征到最后一个特征的尺度以及前一个特征的通道数
    """
    _, C, _, _ = b.size()
    N, _, H, W = a.size()

    b = cus_sample(b, mode="size", factors=(H, W), interpolation=interpolation, align_corners=align_corners)
    # 对 a 进行重塑，新增的维度是通道交互组数，方便对每组的通道维度 C 进行求均值操作。
    a = a.reshape(N, -1, C, H, W).mean(1)
    return b + a


# 对通道进行混洗
def shuffle_channels(x, groups):
    """
    Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,W] -> [N,C,H,W]
    一共C个channel要分成g组混合的channel，先把C reshape成(g, C/g)的形状，
    然后转置成(C/g, g)最后平坦成C组channel
    """
    N, C, H, W = x.size()
    x = x.reshape(N, groups, C // groups, H, W).permute(0, 2, 1, 3, 4)
    return x.reshape(N, C, H, W)


# 对模型的梯度进行裁剪，防止梯度爆炸的情况
# "norm" 表示按梯度的范数裁剪，"value" 表示按梯度的值裁剪
def clip_grad(params, mode, clip_cfg: dict):
    if mode == "norm":
        if "max_norm" not in clip_cfg:
            raise ValueError(f"`clip_cfg` must contain `max_norm`.")
        # norm_type 参数是范数的类型（默认是 2.0，即 L2 范数）。
        # 如果 params 中的梯度的范数超过了 max_norm，它们将被缩放到 max_norm 的大小。
        torch.nn.utils.clip_grad_norm_(
            params, max_norm=clip_cfg.get("max_norm"), norm_type=clip_cfg.get("norm_type", 2.0)
        )
    elif mode == "value":
        if "clip_value" not in clip_cfg:
            raise ValueError(f"`clip_cfg` must contain `clip_value`.")
        # 梯度的每个元素都会被限制在 [-clip_value, clip_value] 范围内。
        torch.nn.utils.clip_grad_value_(params, clip_value=clip_cfg.get("clip_value"))
    else:
        raise NotImplementedError


if __name__ == "__main__":
    a = torch.rand(3, 4, 10, 10)
    b = torch.rand(3, 2, 5, 5)
    print(upsample_reduce(b, a).size())
