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



if __name__ == '__main__':
    # rename_to_seg_dir()
    # modify_mask()
    pass

