#!/bin/sh


/home/huangdn/anaconda/envs/Causal3DNet/bin/python -m src.preprocessing.ROI_data \
	--input /home/huangdn/Causal3D-Net/src/dataset/add_data_finger.xlsx \
	--outdir /home/huangdn/Causal3D-Net/src/dataset \
	--process_num 4

