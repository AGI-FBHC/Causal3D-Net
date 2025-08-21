#!/bin/sh

for i in $(seq 1 10)
do
    TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
    echo "========== Run $i ($TIMESTAMP) =========="
    /home/huangdn/anaconda/envs/Causal3DNet/bin/python -m src.training.train_Causal3DNet \
        --train /home/huangdn/Causal3D-Net/src/dataset/dataset_for_train.xlsx \
        --folds 10 \
        --cuda 5 \
        --outdir /home/huangdn/Causal3D-Net/src/results \
        --weight /home/huangdn/Causal3D-Net/src/results/2025-06-21_06-49-30/last_model.pth
done
