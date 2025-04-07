_base_ = [
    "../_base_/common.py",
    "../_base_/train.py",
    "../_base_/test.py",
]

has_test = True  # 是否包含测试过程
deterministic = True  # 是否使用确定性算法（保证结果可复现）
# 是否使用自定义的 worker 初始化函数
# 自定义的 worker 初始化函数（worker_init_fn） 是指在使用 PyTorch 的 DataLoader 加载数据时
# 为每个数据加载子进程（worker） 设置初始化逻辑的函数
use_custom_worker_init = False
model_name = "MFFN"

# 在 base 的基础上进行覆盖和补充（相当于继承）
train = dict(
    batch_size=8,
    num_workers=4,
    use_amp=True,  # 是否使用混合精度训练（加速 + 节省显存）
    num_epochs=50,
    epoch_based=True,  # 是否按 epoch 调度（否则按 iteration）
    lr=0.05,
    optimizer=dict(
        mode="sgd",
        set_to_none=True,  # 是否将梯度设为 None（节省显存）
        group_mode="finetune",
        cfg=dict(
            momentum=0.9,
            weight_decay=5e-4,
            nesterov=False,
        ),
    ),
    # 学习率调度器 scheduler
    sche_usebatch=True,  # 是否按 batch 更新调度器
    scheduler=dict(  # 预热设置
        warmup=dict(
            num_iters=0,  # 预热迭代次数
            initial_coef=0.01,  # 初始学习率系数
            mode="linear",  # 线性预热
        ),
        mode="f3",  # 调度器模式（f3 是自定义策略名）
        cfg=dict(
            lr_decay=0.9,  # 每轮学习率衰减系数
            min_coef=0.001,  # 最小学习率系数
        ),
    ),
)

test = dict(
    batch_size=8,
    num_workers=4,
    show_bar=False,  # 是否显示进度条
)

datasets = dict(
    train=dict(
        dataset_type="MFFN_cod_tr",
        shape=dict(h=384, w=384),
        path=["cardd_tr"],
        interp_cfg=dict(),  # 插值配置（为空表示默认设置）
    ),
    test=dict(
        dataset_type="MFFN_cod_te",
        shape=dict(h=384, w=384),
#         path=["cpd1k_te", "cod10k_te", "nc4k"],
        path=["cardd_te"],
        interp_cfg=dict(),
    ),
)
