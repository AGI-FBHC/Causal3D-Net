# -*- coding: utf-8 -*-
# @Time    : 2025/4/26 20:37
# @Author  : D.N. Huang
# @Email   : CarlCypress@yeah.net
# @FileName: train.py
# @Project : Causal3D-Net
import torch
import logging
import os, argparse
import nibabel as nib
from torchvision import transforms
from tqdm import tqdm
from dataset.PC_dataset import *
from torch.utils.data import DataLoader


def training(excel_path, output_dir, logging_path):
    batch_size = 1
    learning_rate = 0.001
    num_epochs = 500
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    train_dataset = PCDataset(excel_path=excel_path, transform=transform)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
    )
    records = []
    for _, X, y in tqdm(train_loader):
        _ = _[0]
        X = X[0]
        X_np = X.cpu().numpy()
        shape = X_np.shape
        record = {
            'image_path': _,
            'x': shape[1],
            'y': shape[2],
            'z': shape[0],
        }
        records.append(record)
        # 保存到Excel
    df = pd.DataFrame(records)
    df.to_excel('/home/huangdn/Causal3D-Net/src/logging_record/output.xlsx', index=False)
    print('Saved to output.xlsx')
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Causal 3D Net model training")
    parser.add_argument(
        "--input", type=str,
        default="/home/huangdn/Causal3D-Net/src/data/roi_data_finger.xlsx",
        # required=True,
        help="Excel path for model training image set."
    )
    parser.add_argument(
        "--outdir", type=str,
        default="/home/huangdn/Causal3D-Net/src/results",
        # required=True,
        help="Model weights saving path."
    )
    parser.add_argument(
        "--log_path", type=str,
        default="/home/huangdn/Causal3D-Net/src/logging_record",
        help="Logging record path."
    )
    args = parser.parse_args()
    training(
        args.input,
        args.outdir,
        args.log_path,
    )

