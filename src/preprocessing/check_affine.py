# -*- coding: utf-8 -*-
# @Time    : 2025/3/24 18:46
# @Author  : D.N. Huang
# @Email   : CarlCypress@yeah.net
# @FileName: check_affine.py
# @Project : Causal3D-Net
import os
import numpy as np
import pandas as pd
import nibabel as nib
from tqdm import tqdm



def check_affine():
    dataset_excel = pd.read_excel("/home/huangdn/Causal3D-Net/src/dataset/dataset.xlsx")
    processed_data = []
    for _, row in tqdm(dataset_excel.iterrows(), total=dataset_excel.shape[0], desc="Processing images"):
        image_path = row['image_path']
        mask_path = row['mask_path']
        cancer = row['cancer']
        img = nib.load(image_path)
        depth = img.shape[2]
        affine_reshaped = img.affine[:3, :3].flatten()
        processed_data.append([image_path, mask_path, cancer] + list(affine_reshaped) + [depth])
    columns = ['image_path', 'mask_path', 'cancer'] + [f"affine_{i+1}" for i in range(9)] + ['depth']
    result_df = pd.DataFrame(processed_data, columns=columns)

    result_df.to_excel("/home/huangdn/Causal3D-Net/src/dataset/dataset_affine.xlsx", index=False)
    pass


if __name__ == "__main__":
    check_affine()


