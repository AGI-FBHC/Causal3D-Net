# -*- coding: utf-8 -*-
# @Time    : 2025/5/30 16:24
# @Author  : D.N. Huang
# @Email   : CarlCypress@yeah.net
# @FileName: infer_PANDA.py
# @Project : Causal3D-Net
import torch
import torchio as tio
import torch.nn as nn
import torch.cuda as cuda
import torch.optim as optim
from torch.utils.data import DataLoader
import nibabel as nib
import numpy as np
import os
from tqdm import tqdm

from src.dataset.PC_dataset import PCDataset
from src.augmentation.window import Windowing
from src.models.PANDA import SegNet, MultiTask3DCNN
from src.utils.visual3D import visualize_prediction

import warnings

warnings.filterwarnings("ignore", category=UserWarning)


def infer_stage_1(model_path, excel_path, cuda_id=4):
    device = torch.device(f"cuda:{cuda_id}")
    model = SegNet(mask_num=1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    transform = tio.Compose([
        # pre
        Windowing(window_center=70, window_width=340),
        tio.RescaleIntensity(out_min_max=(0, 1)),
        tio.Resize((40, 160, 256)),
    ])

    infer_dataset = PCDataset(excel_path=excel_path,
                              transform=transform,
                              return_type=1,)
    infer_loader = DataLoader(infer_dataset,
                              batch_size=1,
                              shuffle=False,
                              num_workers=1,
                              pin_memory=True)
    print("start infer...")
    save_root = "/home/huangdn/Causal3D-Net/src/results/2025-05-29_09-15-37"
    with torch.no_grad():
        for i, (x, y) in enumerate(tqdm(infer_loader)):
            x = x.to(device)
            y_sgs, y_cls = model(x)

            if isinstance(y_sgs, list):  # 多尺度输出，取最高分辨率
                y_sgs = y_sgs[0]

            pred_mask = (torch.sigmoid(y_sgs) > 0.5).long().squeeze(1)

            # 把 batch 和 channel 维都去掉
            image_vol = x[0, 0].cpu()  # [D, H, W]
            gt_vol = y[0, 0].cpu()  # [D, H, W]
            pred_vol = pred_mask[0].cpu()  # [D, H, W]

            instance_dir = os.path.join(save_root, f"sample_{i}")
            os.makedirs(instance_dir, exist_ok=True)

            for slice_idx in range(image_vol.shape[0]):
                save_path = os.path.join(instance_dir, f"slice_{slice_idx:03d}.png")
                visualize_prediction(
                    image=image_vol,
                    gt_mask=gt_vol,
                    pred_mask=pred_vol,
                    slice_idx=slice_idx,
                    save_path=save_path,
                )
    pass


if __name__ == '__main__':
    infer_stage_1(
        model_path="/home/huangdn/Causal3D-Net/src/results/2025-05-29_09-15-37/best_model.pth",
        excel_path="/home/huangdn/Causal3D-Net/src/dataset/infer_dataset.xlsx",
    )
    pass
