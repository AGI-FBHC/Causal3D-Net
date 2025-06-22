# -*- coding: utf-8 -*-
# @Time    : 2025/4/26 21:03
# @Author  : D.N. Huang
# @Email   : CarlCypress@yeah.net
# @FileName: dataset_splitting.py
# @Project : Causal3D-Net
import os, argparse
import numpy as np
import pandas as pd
from pprint import pprint
from sklearn.model_selection import train_test_split


def split(excel_path, output_dir):
    all_data = pd.read_excel(excel_path)
    train_data, test_data = train_test_split(
        all_data,
        test_size=0.2,
        random_state=42,  # 保证每次划分一样
        shuffle=True
    )
    train_data = train_data.sort_values(by="image_path", ascending=True)
    test_data = test_data.sort_values(by="image_path", ascending=True)

    train_save_path = os.path.join(output_dir, "train_dataset.xlsx")
    test_save_path = os.path.join(output_dir, "test_dataset.xlsx")

    train_data.to_excel(train_save_path, index=False)
    print(f"Train data saved to {train_save_path}")
    test_data.to_excel(test_save_path, index=False)
    print(f"Test data saved to {test_save_path}")
    pass


def extract_center(path):
    for i in range(1, 20):
        if f"Center{str(i).zfill(2)}" in path:
            return f"center{i}"
    return None


def extract_source(path):
    if "public" in path:
        return "public"
    else:
        return "private"


def split_with_plan(excel_path, output_dir, is_print=False):
    """Divide dataset according to the plan described in the paper."""

    all_data = pd.read_excel(excel_path)

    # 加入 center 和 source 列（假设你有这两个函数）
    all_data["center"] = all_data["image_path"].apply(extract_center)
    all_data["source"] = all_data["image_path"].apply(extract_source)

    # ==== Private Test ====
    private = all_data[all_data["source"] == "private"]

    private_c1 = private[private["center"] == "center1"]
    c1_0 = private_c1[private_c1["cancer"] == 0].sample(n=72, random_state=42)
    c1_1 = private_c1[private_c1["cancer"] == 1].sample(n=32, random_state=42)
    private_c1_test = pd.concat([c1_0, c1_1])

    private_c4_test = private[private["center"] == "center4"]

    private_test = pd.concat([private_c1_test, private_c4_test])
    private_train = private.drop(private_test.index)

    # ==== Public Test ====
    public = all_data[all_data["source"] == "public"]

    # 先将所有用于个别采样的 center 数据单独处理，避免重复
    public_test_parts = []

    # center2 和 center11 全部进入 test（public_c_test）
    public_centers = ["center2", "center11", "center5", "center6", "center8"]
    public_c_test = public[public["center"].isin(public_centers)]
    public_test_parts.append(public_c_test)

    # center4: cancer 1 取 80
    center4 = public[public["center"] == "center4"]
    public_4_test = center4[center4["cancer"] == 1].sample(n=80, random_state=42)
    public_test_parts.append(public_4_test)

    # center12: cancer 1 取 18
    center12 = public[public["center"] == "center12"]
    public_12_test = center12[center12["cancer"] == 1].sample(n=18, random_state=42)
    public_test_parts.append(public_12_test)

    # center14: cancer 0 和 1 各取 17
    center14 = public[public["center"] == "center14"]
    c14_0 = center14[center14["cancer"] == 0].sample(n=17, random_state=42)
    c14_1 = center14[center14["cancer"] == 1].sample(n=17, random_state=42)
    public_14_test = pd.concat([c14_0, c14_1])
    public_test_parts.append(public_14_test)

    # public test 总集
    public_test = pd.concat(public_test_parts)
    public_train = public.drop(public_test.index)

    # ==== 合并总集 ====
    train_df = pd.concat([private_train, public_train]).reset_index(drop=True)
    test_df = pd.concat([private_test, public_test]).reset_index(drop=True)

    if is_print:
        print(f"All size: {len(train_df)}")
        print(f"Train size: {len(train_df)}")
        print(f"Test size: {len(test_df)}")
        print(all_data.shape[0] == len(train_df)+ len(test_df))

    os.makedirs(output_dir, exist_ok=True)

    # 建议保存为 .xlsx 或改函数
    train_save_path = os.path.join(output_dir, "dataset_for_train.xlsx")
    test_save_path = os.path.join(output_dir, "dataset_for_test.xlsx")
    train_df = train_df.iloc[:, :-2]
    test_df = test_df.iloc[:, :-2]
    train_df.to_excel(train_save_path, index=False)
    test_df.to_excel(test_save_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset splitting")
    parser.add_argument(
        "--input", type=str,
        default="/home/huangdn/Causal3D-Net/src/dataset/dataset_for_confounder.xlsx",
        # default="/home/huangdn/Causal3D-Net/src/dataset/dataset_for_roi.xlsx",
        help="Origin sorted images and masks Excel file path."
    )
    parser.add_argument(
        "--outdir", type=str,
        default="/home/huangdn/Causal3D-Net/src/dataset/",
        help="Output folder path."
    )
    args = parser.parse_args()
    split_with_plan(args.input, args.outdir, is_print=True)
