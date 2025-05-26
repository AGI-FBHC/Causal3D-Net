# -*- coding: utf-8 -*-
# @Time    : 2025/4/29 09:25
# @Author  : D.N. Huang
# @Email   : CarlCypress@yeah.net
# @FileName: box_check.py
# @Project : Causal3D-Net
import os
import numpy as np
import pandas as pd
import nibabel as nib
import SimpleITK as sitk
import matplotlib.pyplot as plt
import seaborn as sns


def make_box_mask():
    mask_path = "/home/huangdn/Causal3D-Net/src/data/masks/Center03Mask00088_00005_public.nii.gz"

    mask = sitk.ReadImage(mask_path)
    mask_array = sitk.GetArrayFromImage(mask).astype('float32')
    coords = np.argwhere(mask_array == 1)
    z_min, y_min, x_min = coords.min(axis=0)
    z_max, y_max, x_max = coords.max(axis=0)
    box_mask_array = np.zeros_like(mask_array, dtype='float32')
    box_mask_array[z_min:z_max + 1, y_min:y_max + 1, x_min:x_max + 1] = 1  # box区域设置为1

    box_mask = sitk.GetImageFromArray(box_mask_array)
    box_mask.CopyInformation(mask)  # 保持和原mask一样的spacing、origin、direction等
    save_path = os.path.join("/home/huangdn/Causal3D-Net/src/results", "box_mask.nii.gz")
    sitk.WriteImage(box_mask, save_path)
    print(f"Box mask已保存到: {save_path}")
    pass


def check_statistic():
    excel_path = "/home/huangdn/Causal3D-Net/src/logging_record/orig_mask_volume_info.xlsx"
    df = pd.read_excel(excel_path)
    depths = df['box_depth']
    widths = df['box_width']
    heights = df['box_height']

    sns.set(style="whitegrid")
    plt.figure(figsize=(15, 4))
    for i, (data, title) in enumerate(zip([depths, widths, heights], ['Depth', 'Width', 'Height'])):
        plt.subplot(1, 3, i + 1)
        sns.histplot(data, bins=20, kde=True)
        plt.title(f'{title} Histogram')
        plt.xlabel(title)
        plt.ylabel('Count')
    plt.tight_layout()
    plt.show()

    # 箱线图
    plt.figure(figsize=(12, 4))
    box_data = pd.DataFrame({
        'Depth': depths,
        'Width': widths,
        'Height': heights
    })

    sns.boxplot(data=box_data, orient="h", palette="Set2")
    plt.title('Box Plots for Depth, Width, Height')
    plt.xlabel('Value')
    plt.tight_layout()
    plt.show()
    pass


if __name__ == '__main__':
    check_statistic()
    pass
