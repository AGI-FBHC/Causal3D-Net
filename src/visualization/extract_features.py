# -*- coding: utf-8 -*-
# @Time    : 2025/11/23 21:43
# @Author  : D.N. Huang
# @Email   : CarlCypress@yeah.net
# @FileName: extract_features.py
# @Project : Causal3D-Net
import os, argparse
from tqdm import tqdm

import numpy as np
import pandas as pd

import torch
import torchio as tio
from torch.utils.data import DataLoader

from src.augmentation.window import Windowing
from src.dataset.PC_dataset import PCDataset
from src.models.Causal3DNet import Causal3DNet


def load_feature_and_label(feature_csv, label_excel):
    print(f"\nLoading Features: {feature_csv}")
    df_feat = pd.read_csv(feature_csv)

    print(f"Loading Labels: {label_excel}")
    df_label = pd.read_excel(label_excel)
    df_label["filename"] = df_label["image_path"].apply(lambda x: os.path.basename(x))

    # test_centers = [0, 3, 6, 8, 15, 16, 17]
    # df_label = df_label[df_label["center"].isin(test_centers)]
    # print(f"Label samples after test filtering: {len(df_label)}")

    df = df_feat.merge(df_label, on="filename", how="inner")
    print(f"Matched samples = {len(df)}")

    return df


def extract_features(model_dir: str = "/home/huangdn/Causal3D-Net/src/results/2025-07-04_00-37-32",
                     test_excel: str = "/home/huangdn/Causal3D-Net/src/dataset/dataset_for_test.xlsx",
                     batch_size: int = 4,
                     cuda_id: int = 5):

    output_dir = os.path.join(model_dir, "features")
    # model_path = os.path.join(model_dir, "best_model.pth")
    model_path = os.path.join(model_dir, "last_model.pth")
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(f"cuda:{cuda_id}" if torch.cuda.is_available() else "cpu")

    # 初始化模型
    model = Causal3DNet()
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # 数据
    pre_transform = tio.Compose([
        Windowing(window_center=70, window_width=340),
        tio.RescaleIntensity(out_min_max=(0, 1)),
        tio.Resize((40, 160, 256)),
    ])
    test_dataset = PCDataset(excel_path=test_excel, transform=pre_transform, return_type=5)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 保存特征列表
    features_causal = []
    features_center = []
    features_individual = []
    filenames = []

    # tqdm 添加进度条
    for fname, x, y_cls, _, center, cluster in tqdm(test_loader, desc="Extracting features"):
        x = x.to(device)

        ((y_indi, y_main, y_cent),
         (feat_individual, feat_cls, feat_center)) = model(x)

        features_causal.append(feat_cls.detach().cpu().numpy())
        features_center.append(feat_center.detach().cpu().numpy())
        features_individual.append(feat_individual.detach().cpu().numpy())
        filenames.extend(fname)

    features_causal = np.concatenate(features_causal, axis=0)
    features_center = np.concatenate(features_center, axis=0)
    features_individual = np.concatenate(features_individual, axis=0)

    # 保存为 CSV，每行第一列是文件名
    df_causal = pd.DataFrame(features_causal)
    df_causal.insert(0, "filename", filenames)
    df_causal.to_csv(os.path.join(output_dir, "features_causal.csv"), index=False)

    df_center = pd.DataFrame(features_center)
    df_center.insert(0, "filename", filenames)
    df_center.to_csv(os.path.join(output_dir, "features_center.csv"), index=False)

    df_individual = pd.DataFrame(features_individual)
    df_individual.insert(0, "filename", filenames)
    df_individual.to_csv(os.path.join(output_dir, "features_individual.csv"), index=False)

    print(f"✅ Features saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract Causal3D-Net features on dataset")

    # /home/huangdn/Causal3D-Net/src/results/2025-07-04_00-37-32
    # /home/huangdn/Causal3D-Net/src/results/2025-11-05_02-24-16
    # /home/huangdn/Causal3D-Net/src/results/2025-11-27_13-25-51
    parser.add_argument("--model_dir", type=str,
        default="/home/huangdn/Causal3D-Net/src/results/2025-11-27_13-25-51",
        required=False,
        help="Directory of the trained model")
    parser.add_argument("--test_excel", type=str,
        # default="/home/huangdn/Causal3D-Net/src/dataset/dataset_for_test.xlsx",
        default="/home/huangdn/Causal3D-Net/src/dataset/dataset_for_train.xlsx",
        required=False,
        help="Path to the test Excel file")
    parser.add_argument("--batch_size", type=int, default=4,
        required=False,
        help="Batch size for DataLoader")
    parser.add_argument("--cuda_id", type=int, default=6,
        required=False,
        help="CUDA device ID")
    args = parser.parse_args()

    extract_features(
        model_dir=args.model_dir,
        test_excel=args.test_excel,
        batch_size=args.batch_size,
        cuda_id=args.cuda_id
    )
