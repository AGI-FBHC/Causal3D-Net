# -*- coding: utf-8 -*-
# @Time    : 2025/5/27 10:09
# @Author  : D.N. Huang
# @Email   : CarlCypress@yeah.net
# @FileName: train_PANDA.py
# @Project : Causal3D-Net
import torch
import torchio as tio
import torch.nn as nn
import torch.cuda as cuda
import torch.optim as optim
from torch.utils.data import DataLoader

import time
import logging
import os, argparse
import nibabel as nib
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt

from src.dataset.PC_dataset import PCDataset
from src.models.PANDA import SegNet, MultiTask3DCNN
from src.augmentation.window import Windowing
from src.augmentation.brightness import MultiplicativeBrightnessTransform
from src.augmentation.contrast import ContrastTransform
from src.augmentation.gamma import GammaTransform
from src.augmentation.gaussian_blur import GaussianBlurTransform
from src.augmentation.gaussian_noise import GaussianNoiseTransform
from src.augmentation.low_resolution import SimulateLowResolutionTransform
from src.metric.loss import DiceLoss, MultiScaleSegmentationLoss
from src.lr_scheduler.linear_lr import linear_lr_lambda
from src.utils.plot_metrics import plot_combined_metrics
from src.utils.init_weights import init_weights_kaiming

import warnings

warnings.filterwarnings("ignore", category=UserWarning)


def stage_1_train(train_excel,
                  test_excel,
                  cuda_id=5,
                  output_dir="/home/huangdn/Causal3D-Net/src/results"):
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    current_dir = os.path.join(output_dir, current_time)
    os.makedirs(current_dir, exist_ok=True)
    log_filename = os.path.join(current_dir, "train.log")
    best_model_save_path = os.path.join(current_dir, "best_model.pth")
    last_model_save_path = os.path.join(current_dir, "last_model.pth")
    logging.basicConfig(
        filename=log_filename,
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger().addHandler(console)

    batch_size = 4
    initial_lr = 1e-2
    weight_decay = 3e-5
    num_epochs = 1000
    device = torch.device(f"cuda:{cuda_id}" if torch.cuda.is_available() else "cpu")

    model = SegNet(mask_num=1)
    model.apply(init_weights_kaiming)
    model.to(device)
    criterion = MultiScaleSegmentationLoss()
    optimizer = optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=linear_lr_lambda(num_epochs)
    )

    transform = tio.Compose([
        # pre
        Windowing(window_center=70, window_width=340),
        tio.RescaleIntensity(out_min_max=(0, 1)),
        tio.Resize((40, 160, 256)),

        # aug
        GaussianNoiseTransform(
            noise_variance=(0, 0.1),
            p_per_channel=1.0,
            synchronize_channels=True,
            p=0.1
        ),
        GaussianBlurTransform(
            blur_sigma=(0.5, 1.0),
            synchronize_channels=False,
            synchronize_axes=False,
            p_per_channel=0.5,
            p=0.2
        ),
        MultiplicativeBrightnessTransform(
            multiplier_range=(0.75, 1.25),
            synchronize_channels=False,
            p_per_channel=1.0,
            p=0.15
        ),
        ContrastTransform(
            contrast_range=(0.75, 1.25),
            preserve_range=True,
            synchronize_channels=False,
            p_per_channel=1.0,
            p=0.15
        ),
        SimulateLowResolutionTransform(
            scale=(0.5, 1.0),
            synchronize_channels=False,
            synchronize_axes=True,
            ignore_axes=(0,),
            allowed_channels=None,
            p_per_channel=0.5,
            p=0.25
        ),
        GammaTransform(
            gamma=(0.7, 1.5),
            p_invert_image=1.0,
            synchronize_channels=False,
            p_per_channel=1.0,
            p_retain_stats=1.0,
            p=0.1
        ),
        GammaTransform(
            gamma=(0.7, 1.5),
            p_invert_image=1.0,
            synchronize_channels=False,
            p_per_channel=1.0,
            p_retain_stats=1.0,
            p=0.3
        ),
    ])

    train_dataset = PCDataset(excel_path=train_excel,
                              transform=transform,
                              return_type=1)
    test_dataset = PCDataset(excel_path=test_excel,
                             transform=transform,
                             return_type=1)
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=4,
                              pin_memory=True)
    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=4,
                             pin_memory=True)

    all_train_losses, all_test_losses = [], []
    all_train_dices, all_test_dices = [], []

    best_dice = 0.0
    for epoch in tqdm(range(num_epochs)):
        model.train()
        train_loss, train_dice = 0.0, 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            y_sgs, y_cls = model(x)
            loss = criterion(y_sgs, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            with torch.no_grad():
                pred_mask = torch.sigmoid(y_sgs[0])
                pred_bin = (pred_mask > 0.5).float()
                dice = 1 - DiceLoss()(pred_bin, y)
                train_dice += dice.item()
        avg_train_loss = train_loss / len(train_loader)
        avg_train_dice = train_dice / len(train_loader)

        model.eval()
        test_loss, test_dice = 0.0, 0.0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)

                y_sgs, y_cls = model(x)
                loss = criterion(y_sgs, y)
                test_loss += loss.item()

                pred_mask = torch.sigmoid(y_sgs[0])
                pred_bin = (pred_mask > 0.5).float()
                dice = 1 - DiceLoss()(pred_bin, y)
                test_dice += dice.item()

        avg_test_loss = test_loss / len(test_loader)
        avg_test_dice = test_dice / len(test_loader)

        all_train_losses.append(avg_train_loss)
        all_test_losses.append(avg_test_loss)
        all_train_dices.append(avg_train_dice)
        all_test_dices.append(avg_test_dice)

        current_lr = optimizer.param_groups[0]['lr']
        log_msg = (f"[Epoch {epoch + 1}/{num_epochs}, LR {current_lr:.6f}] "
                f"Train Loss: {avg_train_loss:.4f}, Train Dice: {avg_train_dice:.4f} | "
                f"Test Loss: {avg_test_loss:.4f}, Test Dice: {avg_test_dice:.4f}")
        logging.info(log_msg)
        scheduler.step()

        if avg_test_dice > best_dice:
            best_dice = avg_test_dice
            torch.save(model.state_dict(), best_model_save_path)
            logging.info(f"Best model updated at epoch {epoch+1}, Dice: {best_dice:.4f}")

        torch.save(model.state_dict(), last_model_save_path)

        plot_combined_metrics(
            all_train_losses, all_test_losses,
            all_train_dices, all_test_dices,
            save_dir=current_dir,
            filename='training_process.png'
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PANDA model training")
    parser.add_argument("--train", type=str,
                        default="/home/huangdn/Causal3D-Net/src/dataset/train_dataset.xlsx",
                        # required=True,
                        help="path to training dataset")
    parser.add_argument("--test", type=str,
                        default="/home/huangdn/Causal3D-Net/src/dataset/test_dataset.xlsx",
                        # required=True,
                        help="path to testing dataset")
    parser.add_argument("--cuda", type=int,
                        default=5,
                        required=False,
                        help="index of GPU to use")
    parser.add_argument("--outdir", type=str,
                        default="/home/huangdn/Causal3D-Net/src/results",
                        required=False,
                        help="output directory")
    args = parser.parse_args()
    stage_1_train(
        train_excel=args.train,
        test_excel=args.test,
        cuda_id=args.cuda,
        output_dir=args.outdir,
    )
    pass
