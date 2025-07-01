#!/bin/sh


/home/huangdn/anaconda/envs/Causal3DNet/bin/python -m src.baselines.baseline_entry \
    --train /home/huangdn/Causal3D-Net/src/dataset/dataset_for_train.xlsx \
    --test /home/huangdn/Causal3D-Net/src/dataset/dataset_for_test.xlsx \
    --method hybrid_transformer \
    --cuda_id 5 \
    --outdir /home/huangdn/Causal3D-Net/src/results

