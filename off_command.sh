#!/bin/sh

PYTHON_BIN=/home/huangdn/anaconda/envs/Causal3DNet/bin/python
TRAIN_SCRIPT=src.training.train_Causal3DNet
TRAIN_FILE=/home/huangdn/Causal3D-Net/src/dataset/dataset_for_train.xlsx
OUTDIR=/home/huangdn/Causal3D-Net/src/results
WEIGHT=/home/huangdn/Causal3D-Net/src/results/2025-06-21_06-49-30/last_model.pth

run_job() {
    GPU=$1
    TIMES=$2
    for i in $(seq 1 $TIMES); do
        TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
        echo "========== GPU $GPU | Run $i ($TIMESTAMP) =========="
        $PYTHON_BIN -m $TRAIN_SCRIPT \
            --train $TRAIN_FILE \
            --folds 10 \
            --cuda $GPU \
            --outdir $OUTDIR \
            --weight $WEIGHT
        sleep 10
    done
}

# CUDA3: 2 次
run_job 3 2 &

# 启动前等 5 秒
sleep 10
# CUDA4: 2 次
run_job 4 2 &

# 再等 5 秒
sleep 10
# CUDA5: 5 次
run_job 5 5 &

wait   # 等待所有后台任务完成
echo "所有任务已完成！"
