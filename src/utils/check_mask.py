# -*- coding: utf-8 -*-
# @Time    : 2025/4/18 09:25
# @Author  : D.N. Huang
# @Email   : CarlCypress@yeah.net
# @FileName: check_mask.py
# @Project : Causal3D-Net
import os
import logging
import numpy as np
import pandas as pd
import nibabel as nib
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor


def print_mask_type(file_path):
    nii = nib.load(file_path)
    data = nii.get_fdata()
    info = f"{file_path}: mask type has {np.unique(data)}"
    logging.info(info)
    pass


def mask_volume_box(file_path):
    """查看`file_path`的MASK文件，获取信息包括文件本身shape:(width, height, depth);
    能完全包住MASK的最小box的shape:(width, height, depth)与前面的维度对应;
    统计MASK的体积大小:data中为1的数量;两个对角顶点的(x,y,z)坐标(最大和最小);
    """
    nii = nib.load(file_path)
    data = nii.get_fdata()
    # 1. 获取文件的shape
    shape = data.shape  # (width, height, depth)
    # 2. 统计MASK的体积大小 (即data中为1的数量)
    mask_volume = np.sum(data == 1)
    # 3. 计算最小包围盒的两个对角顶点
    # 这里通过找到非零区域的最小和最大坐标来确定最小包围盒
    non_zero_indices = np.argwhere(data == 1)
    min_corner = non_zero_indices.min(axis=0)
    max_corner = non_zero_indices.max(axis=0)
    # 最小包围盒的shape
    box_shape = max_corner - min_corner + 1
    # 4. 返回最小包围盒的两个对角顶点坐标 (最小和最大坐标)
    min_corner_coords = min_corner.astype(float)  # 转为float类型
    max_corner_coords = max_corner.astype(float)  # 转为float类型
    # 将所有信息整理成一个列表返回
    result = [file_path] + list(shape) + list(box_shape) + [mask_volume] + min_corner_coords.tolist() + max_corner_coords.tolist()
    return result



def through_files():
    excel_path = "/home/huangdn/Causal3D-Net/src/data/data_finger.xlsx"
    # logging.basicConfig(
    #     filename="/home/huangdn/Causal3D-Net/src/logging_record/mask_type.log",
    #     level=logging.INFO,
    #     format="%(asctime)s - %(message)s",
    #     filemode='w'
    # )
    df = pd.read_excel(excel_path)
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
         mask_path = row["mask_path"]
         print_mask_type(mask_path)
    pass
    # results = []
    # with ProcessPoolExecutor(max_workers=4) as executor:  # 设置最大进程数为4
    #     results = list(tqdm(executor.map(mask_volume_box, df["mask_path"]), total=df.shape[0]))
    # # for index, row in tqdm(df.iterrows(), total=df.shape[0]):
    # #     mask_path = row["mask_path"]
    # #     # 获取每个mask的相关数据
    # #     result = mask_volume_box(mask_path)
    # #     # 在结果前面加上文件名
    # #     result_with_filename = [mask_path] + result
    # #     # 将结果添加到results列表中
    # #     results.append(result_with_filename)
    # # 将所有结果转化为DataFrame
    # results_df = pd.DataFrame(results, columns=[
    #     "file_name", "width", "height", "depth", "box_width", "box_height", "box_depth",
    #     "mask_count", "min_x", "min_y", "min_z", "max_x", "max_y", "max_z"
    # ])
    # # 将结果保存为Excel文件
    # output_path = "/home/huangdn/Causal3D-Net/src/logging_record/mask_volume_info.xlsx"
    # results_df.to_excel(output_path, index=False)
    # print(f"Results saved to {output_path}")


if __name__ == '__main__':
    through_files()

