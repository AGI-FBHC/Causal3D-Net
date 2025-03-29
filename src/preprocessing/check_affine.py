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
import seaborn as sns
import matplotlib.pyplot as plt



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


def analyze_affine_distribution():
    df = pd.read_excel("/home/huangdn/Causal3D-Net/src/dataset/dataset_affine.xlsx")
    z_spacings = df["affine_9"]

    # 计算基本统计描述
    descriptive_stats = z_spacings.describe()
    print("Descriptive statistics:")
    print(descriptive_stats)

    # 绘制直方图
    plt.figure(figsize=(10, 6))
    sns.histplot(z_spacings, kde=True, color='blue', bins=10)
    plt.title('Distribution of Z Spacings')
    plt.xlabel('Z Spacings')
    plt.ylabel('Frequency')
    plt.show()

    # 绘制箱线图
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=z_spacings, color='lightgreen')
    plt.title('Boxplot of Z Spacings')
    plt.xlabel('Z Spacings')
    plt.show()

    # 输出其他可能需要的特征
    print("\nVariance:", z_spacings.var())
    print("Skewness:", z_spacings.skew())
    print("Kurtosis:", z_spacings.kurt())
    pass


def change_mask_affine():
    df = pd.read_excel("/home/huangdn/Causal3D-Net/src/dataset/dataset.xlsx")
    for index, row in df.iterrows():
        image_path = row['image_path']
        mask_path = row['mask_path']
        image_nii = nib.load(image_path)
        mask_nii = nib.load(mask_path)
        updated_mask_nii = nib.Nifti1Image(mask_nii.get_fdata(), image_nii.affine, header=image_nii.header)
        output_mask_path = mask_path.replace(".nii.gz", "_updated.nii.gz")
        nib.save(updated_mask_nii, mask_path)
        break
    pass


if __name__ == "__main__":
    change_mask_affine()


