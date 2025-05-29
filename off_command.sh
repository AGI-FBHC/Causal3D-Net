#!/bin/sh


/home/huangdn/anaconda/envs/Causal3DNet/bin/python -m src.training.train_PANDA \
    --train /home/huangdn/Causal3D-Net/src/dataset/train_dataset.xlsx \
    --test /home/huangdn/Causal3D-Net/src/dataset/test_dataset.xlsx \
    --outdir /home/huangdn/Causal3D-Net/src/results

