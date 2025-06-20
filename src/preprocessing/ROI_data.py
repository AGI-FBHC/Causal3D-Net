# -*- coding: utf-8 -*-
# @Time    : 2025/4/27 10:44
# @Author  : D.N. Huang
# @Email   : CarlCypress@yeah.net
# @FileName: ROI_data.py
# @Project : Causal3D-Net
import os, argparse
import pandas as pd
import torch
from tqdm import tqdm
import numpy as np
import SimpleITK as sitk
from concurrent.futures import ProcessPoolExecutor, as_completed


def process_single_case(index, row):
    image_path = row['image_path']
    mask_path = row['mask_path']
    image_save_path = image_path.replace(".nii.gz", ".npy")
    mask_save_path = mask_path.replace(".nii.gz", ".npy")

    image = sitk.ReadImage(image_path)
    image_array = sitk.GetArrayFromImage(image).astype('float32')
    mask = sitk.ReadImage(mask_path)
    mask_array = sitk.GetArrayFromImage(mask).astype('float32')

    coords = np.argwhere(mask_array == 1)
    if coords.shape[0] == 0:
        raise ValueError(f"No positive region found in mask for index {index}.")

    z_min, y_min, x_min = coords.min(0)
    z_max, y_max, x_max = coords.max(0)

    cropped_image = image_array[z_min:z_max + 1, y_min:y_max + 1, x_min:x_max + 1]
    cropped_mask = mask_array[z_min:z_max + 1, y_min:y_max + 1, x_min:x_max + 1]

    np.save(image_save_path, cropped_image)
    np.save(mask_save_path, cropped_mask)

    return index, os.path.basename(image_path), cropped_image.shape


def crop_data(excel_path, save_path, num_workers=4):
    roi_data_finger_save_path = os.path.join(save_path, "dataset_for_roi.xlsx")
    dataset_excel = pd.read_excel(excel_path)

    futures = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for index, row in dataset_excel.iterrows():
            futures.append(executor.submit(process_single_case, index, row))

        for future in tqdm(as_completed(futures), total=len(futures), desc="Cropping"):
            try:
                index, image_name, shape = future.result()
                print(f"Finished cropping {image_name}, shape = {shape}.")
            except Exception as e:
                print(f"Error occurred: {e}")

    # 修改并保存新的Excel
    dataset_excel['image_path'] = dataset_excel['image_path'].apply(lambda x: x.replace(".nii.gz", ".npy"))
    dataset_excel['mask_path'] = dataset_excel['mask_path'].apply(lambda x: x.replace(".nii.gz", ".npy"))
    dataset_excel.to_excel(roi_data_finger_save_path, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Crop images and masks")
    parser.add_argument(
        "--input", type=str,
        default="/home/huangdn/Causal3D-Net/src/dataset/dataset.xlsx",
        # required=True,
        help="Path to excel file containing images and masks.",
    )
    parser.add_argument(
        "--outdir", type=str,
        default="/home/huangdn/Causal3D-Net/src/dataset",
        # required=True,
        help="Output roi data finger save path."
    )
    parser.add_argument(
        "--process_num",
        type=int,
        default=16,
        help="Number of concurrent processes to run, be careful not to exceed the number of CPU cores."
    )
    args = parser.parse_args()
    crop_data(
        args.input,
        args.outdir,
        args.process_num,
    )
