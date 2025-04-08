has_test = True
base_seed = 0
deterministic = True
log_interval = dict(txt=20, tensorboard=0)
load_from = ''
resume_from = ''
model_name = 'MFFN'
experiment_tag = 'demo'
train = dict(
    batch_size=8,
    num_workers=4,
    use_amp=True,
    num_epochs=50,
    num_iters=30000,
    epoch_based=True,
    lr=0.05,
    optimizer=dict(
        mode='sgd',
        set_to_none=True,
        group_mode='finetune',
        cfg=dict(momentum=0.9, weight_decay=0.0005, nesterov=False)),
    grad_acc_step=1,
    sche_usebatch=True,
    scheduler=dict(
        warmup=dict(num_iters=0, initial_coef=0.01, mode='linear'),
        mode='f3',
        cfg=dict(lr_decay=0.9, min_coef=0.001)),
    save_num_models=1,
    ms=dict(enable=False, extra_scales=[0.75, 1.25, 1.5]),
    grad_clip=dict(enable=False, mode='value', cfg=dict()),
    ema=dict(
        enable=False, cmp_with_origin=True, force_cpu=False, decay=0.9998))
test = dict(
    batch_size=8,
    num_workers=4,
    eval_func='default_test',
    clip_range=None,
    tta=dict(
        enable=False,
        reducation='mean',
        cfg=dict(
            HorizontalFlip=dict(),
            VerticalFlip=dict(),
            Rotate90=dict(angles=[0, 90, 180, 270]),
            Scale=dict(
                scales=[0.75, 1, 1.5],
                interpolation='bilinear',
                align_corners=False),
            Add=dict(values=[0, 10, 20]),
            Multiply=dict(factors=[1, 2, 5]),
            FiveCrops=dict(crop_height=224, crop_width=224),
            Resize=dict(
                sizes=[0.75, 1, 1.5],
                original_size=224,
                interpolation='bilinear',
                align_corners=False))),
    show_bar=False)
use_custom_worker_init = False
datasets = dict(
    train=dict(
        dataset_type='MFFN_cod_tr',
        shape=dict(h=384, w=384),
        path=dict(
            cardd_tr=dict(
                root='/data1_ssd/gyy/CarDD/data/CarDD_SOD/CarDD-TR',
                image=dict(path='CarDD-TR-Image', suffix='.jpg'),
                mask=dict(path='CarDD-TR-Mask', suffix='.png'),
                edge=dict(path='CarDD-TR-Edge', suffix='.png'))),
        interp_cfg=dict()),
    test=dict(
        dataset_type='MFFN_cod_te',
        shape=dict(h=384, w=384),
        path=dict(
            cardd_te=dict(
                root='/data1_ssd/gyy/CarDD/data/CarDD_SOD/CarDD-TE',
                image=dict(path='CarDD-TE-Image', suffix='.jpg'),
                mask=dict(path='CarDD-TE-Mask', suffix='.png'),
                edge=dict(path='CarDD-TE-Edge', suffix='.png'))),
        interp_cfg=dict()))
use_ddp = False
proj_root = '/data1_ssd/gyy/CarDD/code/SOD/MFFN'
exp_name = 'MFFN_BS8_LR0.05_E50_H384_W384_OPMsgd_OPGMfinetune_SCf3_AMP_INFOdemo'
output_dir = '/data1_ssd/gyy/CarDD/code/SOD/MFFN/output'
path = dict(
    output_dir='/data1_ssd/gyy/CarDD/code/SOD/MFFN/output',
    pth_log=
    '/data1_ssd/gyy/CarDD/code/SOD/MFFN/output/MFFN_BS8_LR0.05_E50_H384_W384_OPMsgd_OPGMfinetune_SCf3_AMP_INFOdemo',
    tb=
    '/data1_ssd/gyy/CarDD/code/SOD/MFFN/output/MFFN_BS8_LR0.05_E50_H384_W384_OPMsgd_OPGMfinetune_SCf3_AMP_INFOdemo/tb',
    save=
    '/data1_ssd/gyy/CarDD/code/SOD/MFFN/output/MFFN_BS8_LR0.05_E50_H384_W384_OPMsgd_OPGMfinetune_SCf3_AMP_INFOdemo/pre',
    pth=
    '/data1_ssd/gyy/CarDD/code/SOD/MFFN/output/MFFN_BS8_LR0.05_E50_H384_W384_OPMsgd_OPGMfinetune_SCf3_AMP_INFOdemo/pth',
    final_full_net=
    '/data1_ssd/gyy/CarDD/code/SOD/MFFN/output/MFFN_BS8_LR0.05_E50_H384_W384_OPMsgd_OPGMfinetune_SCf3_AMP_INFOdemo/pth/checkpoint_final.pth',
    final_state_net=
    '/data1_ssd/gyy/CarDD/code/SOD/MFFN/output/MFFN_BS8_LR0.05_E50_H384_W384_OPMsgd_OPGMfinetune_SCf3_AMP_INFOdemo/pth/state_final.pth',
    tr_log=
    '/data1_ssd/gyy/CarDD/code/SOD/MFFN/output/MFFN_BS8_LR0.05_E50_H384_W384_OPMsgd_OPGMfinetune_SCf3_AMP_INFOdemo/tr_2025-04-07.txt',
    te_log=
    '/data1_ssd/gyy/CarDD/code/SOD/MFFN/output/MFFN_BS8_LR0.05_E50_H384_W384_OPMsgd_OPGMfinetune_SCf3_AMP_INFOdemo/te_2025-04-07.txt',
    trans_log=
    '/data1_ssd/gyy/CarDD/code/SOD/MFFN/output/MFFN_BS8_LR0.05_E50_H384_W384_OPMsgd_OPGMfinetune_SCf3_AMP_INFOdemo/trans_2025-04-07.txt',
    cfg_copy=
    '/data1_ssd/gyy/CarDD/code/SOD/MFFN/output/MFFN_BS8_LR0.05_E50_H384_W384_OPMsgd_OPGMfinetune_SCf3_AMP_INFOdemo/cfg_2025-04-07 19:38:03.707873.py',
    excel=
    '/data1_ssd/gyy/CarDD/code/SOD/MFFN/output/MFFN_BS8_LR0.05_E50_H384_W384_OPMsgd_OPGMfinetune_SCf3_AMP_INFOdemo/results.xlsx',
    trainer_copy=
    '/data1_ssd/gyy/CarDD/code/SOD/MFFN/output/MFFN_BS8_LR0.05_E50_H384_W384_OPMsgd_OPGMfinetune_SCf3_AMP_INFOdemo/trainer_2025-04-07 19:38:03.707876.txt'
)
