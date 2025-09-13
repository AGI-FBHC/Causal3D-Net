# -*- coding: utf-8 -*-
# @Time    : 2025/9/2 21:52
# @Author  : D.N. Huang
# @Email   : CarlCypress@yeah.net
# @FileName: medalab_end_to_end_workflow.py
# @Project : Causal3D-Net

# =============================================================================
# recurrence: https://github.com/medalab-dladpcfncp/end_to_end_workflow
# =============================================================================

import os
import shutil
import subprocess
import numpy as np
import pandas as pd
from tqdm import tqdm
import nibabel as nib


def rename_to_seg_dir(excel_path="/home/huangdn/Causal3D-Net/src/dataset/dataset.xlsx",
                      dst_dir="/home/huangdn/U-Mamba/data/nnUNet_raw/Dataset006_pancreasTumour",
                      map_save_path="/home/huangdn/Causal3D-Net/src/logging_record/seg_file_map.xlsx"):
    excel = pd.read_excel(excel_path)
    df = excel[["image_path"]]
    df["cancer"] = excel["cancer"]
    df["seg_image_path"] = [
        os.path.join(dst_dir, "imagesTs", f"pancreas_{i:04d}_0000.nii.gz") for i in range(1, len(df)+1)
    ]
    df["seg_mask_path"] = [
        os.path.join(dst_dir, "infer", f"pancreas_{i:04d}.nii.gz") for i in range(1, len(df)+1)
    ]
    for src, dst in tqdm(zip(df["image_path"], df["seg_image_path"]), total=len(df), desc="Copying"):
        shutil.copy(src, dst)
    df.to_excel(map_save_path, index=False)
    pass


def modify_mask(seg_map_path="/home/huangdn/Causal3D-Net/src/logging_record/seg_file_map.xlsx"):
    map_file = pd.read_excel(seg_map_path)
    for idx, row in tqdm(map_file.iterrows(), total=len(map_file), desc="Modifying masks"):
        seg_mask_path = row["seg_mask_path"]
        cancer = row["cancer"]
        print(seg_mask_path, cancer)
        if cancer == 0:
            img = nib.load(seg_mask_path)
            data = img.get_fdata()
            data[:] = 0
            new_img = nib.Nifti1Image(data, img.affine, img.header)
            nib.save(new_img, seg_mask_path)
            pass
        pass
    pass


def move_files(df, img_col, mask_col, target_img_dir, target_mask_dir):
    for _, row in df.iterrows():
        img_src = row[img_col]
        mask_src = row[mask_col]

        img_dst = os.path.join(target_img_dir, os.path.basename(img_src))
        mask_dst = os.path.join(target_mask_dir, os.path.basename(mask_src))

        if os.path.exists(img_src):
            shutil.move(img_src, img_dst)
        else:
            print(f"⚠️ 图像文件不存在: {img_src}")
        if os.path.exists(mask_src):
            shutil.move(mask_src, mask_dst)
        else:
            print(f"⚠️ 掩膜文件不存在: {mask_src}")


def retraining_model(train_excel_path="/home/huangdn/Causal3D-Net/src/dataset/dataset_for_train.xlsx",
                     test_excel_path="/home/huangdn/Causal3D-Net/src/dataset/dataset_for_test.xlsx",
                     map_excel_path="/home/huangdn/Causal3D-Net/src/logging_record/seg_file_map.xlsx",
                     train_img_dir="/home/huangdn/U-Mamba/data/nnUNet_raw/Dataset008_pancreasTumour/imagesTr",
                     train_mask_dir="/home/huangdn/U-Mamba/data/nnUNet_raw/Dataset008_pancreasTumour/labelsTr",
                     test_img_dir="/home/huangdn/U-Mamba/data/nnUNet_raw/Dataset008_pancreasTumour/imagesTs",
                     test_mask_dir="/home/huangdn/U-Mamba/data/nnUNet_raw/Dataset008_pancreasTumour/ground_truth"):
    train_excel = pd.read_excel(train_excel_path)
    test_excel = pd.read_excel(test_excel_path)
    map_excel = pd.read_excel(map_excel_path)

    train_excel["image_path"] = train_excel["image_path"].str.replace(".npy", ".nii.gz", regex=False)
    test_excel["image_path"] = test_excel["image_path"].str.replace(".npy", ".nii.gz", regex=False)

    train_paths = set(train_excel["image_path"])
    test_paths = set(test_excel["image_path"])

    map_train = map_excel[map_excel["image_path"].isin(train_paths)].copy()
    map_test = map_excel[map_excel["image_path"].isin(test_paths)].copy()

    move_files(map_train, "seg_image_path", "seg_mask_path", train_img_dir, train_mask_dir)
    move_files(map_test, "seg_image_path", "seg_mask_path", test_img_dir, test_mask_dir)


if __name__ == '__main__':
    # rename_to_seg_dir()
    # modify_mask()
    retraining_model()
    pass

