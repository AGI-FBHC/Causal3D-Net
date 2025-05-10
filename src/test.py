# -*- coding: utf-8 -*-
# @Time    : 2025/3/24 18:42
# @Author  : D.N. Huang
# @Email   : CarlCypress@yeah.net
# @FileName: test.py
# @Project : Causal3D-Net
import pandas as pd


data_finger_path = "/home/huangdn/Causal3D-Net/src/dataset/data_finger.xlsx"
roi_data_finger_path = "/home/huangdn/Causal3D-Net/src/dataset/roi_data_finger.xlsx"
df = pd.read_excel(data_finger_path)
df['image_path'] = df['image_path'].str.replace('.nii.gz', '.npy', regex=False)
df['mask_path'] = df['mask_path'].str.replace('.nii.gz', '.npy', regex=False)
df.to_excel(roi_data_finger_path, index=False)