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
from datetime import datetime

from src.metric.compute_score import evaluate_test_result


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
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Moving files"):
        img_src = row[img_col]
        mask_src = row[mask_col]

        img_dst = os.path.join(target_img_dir, os.path.basename(img_src))
        mask_dst = os.path.join(target_mask_dir, os.path.basename(mask_src))

        if os.path.exists(img_src):
            shutil.move(img_src, img_dst)
            pass
        else:
            print(f"⚠️ 图像文件不存在: {img_src}")
        if os.path.exists(mask_src):
            shutil.move(mask_src, mask_dst)
            pass
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
    pass


def dice_coefficient(mask1, mask2):
    mask1 = mask1.astype(bool)
    mask2 = mask2.astype(bool)

    intersection = np.logical_and(mask1, mask2).sum()
    size1 = mask1.sum()
    size2 = mask2.sum()

    if size1 + size2 == 0:
        return 1.0
    return 2. * intersection / (size1 + size2)


def compute_dice(row):
    gt_img = nib.load(row["seg_ground_truth"]).get_fdata()
    infer_img = nib.load(row["seg_infer"]).get_fdata()
    return dice_coefficient(gt_img, infer_img)


def metric_evaluation_for_chen(map_excel_path="/home/huangdn/Causal3D-Net/src/logging_record/seg_file_map.xlsx",
                               test_excel_path="/home/huangdn/Causal3D-Net/src/dataset/dataset_for_test.xlsx",
                               output_dir="/home/huangdn/Causal3D-Net/src/results",):
    test_excel = pd.read_excel(test_excel_path)
    map_excel = pd.read_excel(map_excel_path)
    test_excel["image_path"] = test_excel["image_path"].str.replace(".npy", ".nii.gz", regex=False)

    merged = pd.merge(
        test_excel,
        map_excel[["image_path", "seg_mask_path"]],
        left_on="image_path",
        right_on="image_path",
        how="left"
    )
    merged["seg_mask_path"] = merged["seg_mask_path"].str.replace("Dataset006_pancreasTumour",
                                                                  "Dataset008_pancreasTumour",
                                                                  regex=False)
    merged["seg_ground_truth"] = merged["seg_mask_path"].str.replace("infer",
                                                                     "ground_truth",
                                                                     regex=False)
    merged = merged.rename(columns={"seg_mask_path": "seg_infer"})
    seg_result = merged[["seg_ground_truth", "seg_infer", "cancer", "center"]]
    seg_result["y_prob"] = seg_result.apply(compute_dice, axis=1)
    seg_result["y_pred"] = seg_result["y_prob"] > 0.5
    seg_result = seg_result[["cancer", "center", "y_prob", "y_pred"]]
    metrics = evaluate_test_result(seg_result)

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    current_dir = os.path.join(output_dir, current_time)
    os.makedirs(current_dir, exist_ok=True)
    save_path = os.path.join(current_dir, "chen2.csv")
    metrics.to_csv(save_path, index=False)





if __name__ == '__main__':
    # rename_to_seg_dir()
    # modify_mask()
    # retraining_model()
    metric_evaluation_for_chen()
    pass

