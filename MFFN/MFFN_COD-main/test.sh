#!/usr/bin/env bash
set -e          # 只要脚本发生错误就停止运行
set -u          # 如果遇到不存在的变量就报错并停止执行
set -x          # 运行指令结果的时候，输出对应的指令
set -o pipefail # 确保只要一个子命令失败，整个管道命令就失败

# bash train.sh 2 -在2号 GPU 上运行
export CUDA_VISIBLE_DEVICES="$1"
echo 'Excute the script on GPU: ' "$1"

echo 'For COD'
python test.py --config ./configs/MFFN/MFFN_R50.py \
    --model-name MFFN \
    --batch-size 22 \
    --load-from /data1_hdd/gyy/CarDD/cp/MFFN/model_final.pth \
    --save-path /data1_hdd/gyy/CarDD/results/SAM2-UNet/test_mask

echo 'For SOD'
python test.py --config ./configs/MFFN/MFFN_R50.py \
    --model-name MFFN \
    --batch-size 22 \
    --load-from /data1_hdd/gyy/CarDD/cp/MFFN/model_final.pth \
    --save-path /data1_hdd/gyy/CarDD/results/SAM2-UNet/test_mask
