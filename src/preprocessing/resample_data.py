# -*- coding: utf-8 -*-
# @Time    : 2025/3/29 10:08
# @Author  : D.N. Huang
# @Email   : CarlCypress@yeah.net
# @FileName: resample_data.py
# @Project : Causal3D-Net
import numpy as np
import pandas as pd
import os, argparse
import nibabel as nib
from scipy.ndimage import zoom


def resample_z_direction(
        nii_path: str,
        is_mask: bool = False,
        output_file: str = 'output.nii.gz',
        spacing: float = 1.0,
        target_depth: int = None,
        is_print: bool = False
        ) -> int:
    """对 NIfTI 文件的 z 方向重采样，若 target_depth 不为 None，
    则根据 target_depth 变换厚度，否则根据物理距离 spacing.
    :param nii_path: 输入的 nifti 文件路径。
    :param is_mask: 是否是掩码文件?
    :param output_file: 重采样后保存的文件路径。
    :param spacing: z 轴的新体素间距(单位：mm)。
    :param target_depth: z轴采样目标厚度(单位：pixel)。
    :param is_print: 是否打印文件处理状态?
    :return: 原始影像z轴的厚度(方便逆操作)。
    """
    nii = nib.load(nii_path)
    data = nii.get_fdata()
    affine = nii.affine
    z_spacing = np.abs(affine[2, 2])
    original_depth = data.shape[2]

    if target_depth is not None:
        # Calculate the zoom factor to match the target depth
        zoom_factor = target_depth / original_depth
        spacing = z_spacing / zoom_factor  # Update spacing based on the target depth
    else:
        # Use the provided physical spacing
        zoom_factor = z_spacing / spacing

    scale_factors = [1, 1, zoom_factor]
    new_data = zoom(data, scale_factors, order=0 if is_mask else 3)  # 对image数据采用3次插值，mask数据采用最近邻插值
    new_data = np.rint(new_data).astype(np.uint8) if is_mask else new_data  # mask 可能存在差值后的精度问题，需要舍入

    new_affine = affine.copy()
    new_affine[2, 2] = np.sign(affine[2, 2]) * spacing  # Update the z-spacing in the affine matrix

    new_img = nib.Nifti1Image(new_data, affine=new_affine, header=nii.header)

    # 更新 z 轴的相关字段
    new_img.header['dim'][3] = new_data.shape[2]  # 更新 z 方向的维度
    new_img.header['pixdim'][3] = spacing  # 更新 z 轴体素间距
    new_img.header['srow_z'] = new_affine[2, :4]  # 更新 srow_z（仿射矩阵的第 3 行）

    nib.save(new_img, output_file)

    print(f'resampling {nii_path} completed.') if is_print else None
    return original_depth


def get_z_spacing_list(resample_num):
    if resample_num == 1:
        return list()
    elif resample_num == 3:
        return [1, 3, 5]
    elif resample_num == 5:
        return [1, 2, 3, 4, 5]
    pass


def find_closest_number(lst, num):
    return min(lst, key=lambda x: (abs(x - num), x))


def resample_data():
    parser = argparse.ArgumentParser(description="Resample image and mask")
    parser.add_argument("--excel_path", type=str, default="/home/huangdn/Causal3D-Net/src/dataset/dataset.xlsx", help="Origin sorted images and masks excel file path.")
    parser.add_argument("--out_path", type=str, default="/home/huangdn/Causal3D-Net/src/data", help="Output resampled images and masks dir path.")
    # parser.add_argument("--excel_path", type=str, required=True, help="Origin sorted images and masks Excel file path.")
    # parser.add_argument("--out_path", type=str, required=True, help="Output resampled images and masks dir path.")
    parser.add_argument("--resample_num", type=int, choices=[1, 3, 5], default=5, help="Total after resampling. Choose from 1, 3, or 5.")
    args = parser.parse_args()

    dataset_excel = pd.read_excel(args.excel_path)
    images_save_path = os.path.join(args.out_path, "images")
    masks_save_path = os.path.join(args.out_path, "masks")
    os.makedirs(images_save_path, exist_ok=True)
    os.makedirs(masks_save_path, exist_ok=True)
    z_spacing_list = get_z_spacing_list(args.resample_num)
    for index, row in dataset_excel.iterrows():
        image_path = row['image_path']
        mask_path = row['mask_path']
        cancer = row['cancer']
        image_nii = nib.load(image_path)
        mask_nii = nib.load(mask_path)
        origin_spacing = abs(image_nii.affine[2, 2])
        print(origin_spacing, find_closest_number(z_spacing_list, origin_spacing))
        if index == 1:
            break

    # print(f"{images_save_path} ===== {masks_save_path}")
    pass


if __name__ == "__main__":
    resample_data()

