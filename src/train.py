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
import torchio as tio
from models.ResNet import *
import torch.cuda as cuda
from utils.window import *


def training(excel_path, output_dir, logging_path):
    batch_size = 4
    learning_rate = 0.001
    num_epochs = 500
    device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
    # 构建模型并移到设备上
    model = generate_model(10, n_input_channels=1, n_classes=2).to(device)
    model.eval()  # 推理模式
    transform = tio.Compose([
        Windowing(window_center=70, window_width=340),
        tio.RescaleIntensity(out_min_max=(0, 1)),  # 归一化到 0-1，可选
        tio.Resize((128, 256, 256)),  # 关键这一步！
    ])
    train_dataset = PCDataset(excel_path=excel_path, transform=transform)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
    )
    for _, X, y in tqdm(train_loader):
        print(X.shape)
        print("=" * 10)
        X = X.to(device)
        y_hat = model(X)
        print(y_hat.shape)
        break


    # input_tensor = torch.randn(4, 1, 128, 256, 256).to(device)
    #
    # # 清空缓存并重置峰值统计
    # cuda.empty_cache()
    # cuda.reset_peak_memory_stats(device)
    #
    # with torch.no_grad():
    #     output = model(input_tensor)
    #
    # # 打印推理期间的峰值显存使用
    # used_mem_MB = cuda.max_memory_allocated(device) / 1024 / 1024
    # print(f"[推理] 显存峰值使用: {used_mem_MB:.2f} MB")
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
        args.log_path
    )

