# -*- coding: utf-8 -*-
# @Time    : 2025/11/28 10:49
# @Author  : D.N. Huang
# @Email   : CarlCypress@yeah.net
# @FileName: gcam_for_ct.py
# @Project : Causal3D-Net
import os, argparse

import cv2
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from skimage import measure

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import torchio as tio

from src.dataset.PC_dataset import PCDataset
from src.augmentation.window import Windowing
from src.utils.init_weights import init_weights_kaiming
from src.models.Causal3DNet import PlainConvEncoder, UNetDecoder, ChannelAttentionDecoder


class Causal3DNetForGCAM(nn.Module):
    def __init__(self, mask_num=2, class_num=2, feature_num=256, groups=24):
        super().__init__()
        self.encoder = PlainConvEncoder()
        self.seg_decoder = UNetDecoder(mask_num=mask_num)
        self.individual_stream = ChannelAttentionDecoder(channel=1920, groups=groups, feature_num=feature_num)
        self.cls_stream = ChannelAttentionDecoder(channel=1920, groups=groups, feature_num=feature_num)
        self.center_stream = ChannelAttentionDecoder(channel=1920, groups=groups, feature_num=feature_num)

        self.indi_cls = nn.Linear(feature_num * 2, class_num)
        self.main_cls = nn.Linear(feature_num, class_num)
        self.cent_cls = nn.Linear(feature_num * 2, class_num)

        self.apply(init_weights_kaiming)

    def forward(self, x):
        skip = self.encoder(x)
        _, shared_features = self.seg_decoder(skip)

        individual_confounder = self.individual_stream(skip, shared_features)
        classify_feature = self.cls_stream(skip, shared_features)
        center_confounder = self.center_stream(skip, shared_features)

        y_indi = self.indi_cls(torch.cat([classify_feature, individual_confounder], dim=1))
        y_main = self.main_cls(classify_feature)
        y_cent = self.cent_cls(torch.cat([classify_feature, center_confounder], dim=1))

        return y_main, shared_features[1]


def grad_cam_3d(model, input_tensor, target_class=None):
    model.eval()
    input_tensor.requires_grad = True

    features_grad = {}

    def save_grad(grad):
        features_grad['grad'] = grad

    y_main, features = model(input_tensor)  # features: [B, C, D', H', W']
    features.register_hook(save_grad)

    if target_class is None:
        target_class = y_main.argmax(dim=1)

    loss = 0
    for i in range(input_tensor.size(0)):
        loss += y_main[i, target_class[i]]
    loss = loss / input_tensor.size(0)

    model.zero_grad()
    loss.backward()

    grads = features_grad['grad']  # [B, C, D', H', W']
    weights = grads.mean(dim=(2, 3, 4), keepdim=True)
    cam = F.relu((weights * features).sum(dim=1, keepdim=True))

    cam_min = cam.view(cam.size(0), -1).min(dim=1)[0][:, None, None, None, None]
    cam_max = cam.view(cam.size(0), -1).max(dim=1)[0][:, None, None, None, None]
    cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
    return cam


def upsample_cam_to_input(cam, target_size):
    """
    cam: [B, 1, D', H', W']
    target_size: (D, H, W)
    """
    cam_up = F.interpolate(cam, size=target_size, mode='trilinear', align_corners=False)
    return cam_up  # [B, 1, D, H, W]


def show_slices_with_cam(volume: torch.Tensor, cam: torch.Tensor, mask: torch.Tensor = None,
                         slice_index: int = None, save_path: str = None):
    """
    可视化 volume 和 cam 的某一层 slice。
    Args:
        volume: [1,1,D,H,W]
        cam:    [1,1,D,H,W]
        slice_index: 想看的 D 方向的切片编号（0~D-1）。
                     若为 None，默认展示中间层。
        save_path: 保存路径（可选）
    """
    # 去掉 batch 和 channel
    vol_np = volume.squeeze().detach().cpu().numpy()  # [D, H, W]
    cam_np = cam.squeeze().detach().cpu().numpy()     # [D, H, W]

    D, H, W = vol_np.shape

    slice_index = D // 2 if slice_index is None else max(0, min(slice_index, D - 1))

    img_slice = vol_np[slice_index]
    cam_slice = cam_np[slice_index]

    img_slice = (img_slice - img_slice.min()) / (img_slice.max() - img_slice.min() + 1e-8)

    plt.figure(figsize=(10, 5))
    plt.imshow(img_slice, cmap='gray')
    plt.imshow(cam_slice, cmap='jet', alpha=0.5)

    if mask is not None:
        mask_np = mask.squeeze().detach().cpu().numpy()  # [D,H,W]
        mask_slice = mask_np[slice_index]

        contours = measure.find_contours(mask_slice, level=0.5)

        for contour in contours:
            plt.plot(contour[:, 1], contour[:, 0], linewidth=1, color='red')

    plt.axis('off')

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', dpi=600)

    plt.show()
    plt.close()


def run_gcam_visualization(model_dir: str, excel_path: str, cuda_id: int = 5):
    """
    运行 GCAM 可视化主流程：
    加载模型 -> 对每个样本计算 CAM -> 保存所有切片
    """
    save_root = os.path.join(model_dir, "gcam")

    device = torch.device(f"cuda:{cuda_id}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = Causal3DNetForGCAM().to(device)

    checkpoint_path = os.path.join(model_dir, "best_model.pth")
    print(f"Loading checkpoint from {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    pre_transform = tio.Compose([
        Windowing(window_center=70, window_width=340),
        tio.RescaleIntensity(out_min_max=(0, 1)),
        tio.Resize((40, 160, 256)),
    ])

    dataset = PCDataset(
        excel_path=excel_path,
        transform=pre_transform,
        return_type=5,
    )

    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

    for filename, X, cls_label, msk_label, center, cluster in loader:

        X = X.to(device)
        msk_label = msk_label.to(device)

        print(f"\nProcessing: {filename[0]} ...")

        cam = grad_cam_3d(model, X)
        cam_up = upsample_cam_to_input(cam, target_size=X.shape[2:])

        case_name = filename[0].split(".")[0]
        fig_save_dir = os.path.join(save_root, case_name)
        os.makedirs(fig_save_dir, exist_ok=True)

        depth = X.shape[2]

        # save_path = os.path.join(fig_save_dir, f"gcam_slice_{depth // 2}.png")
        # show_slices_with_cam(volume=X, cam=cam_up, mask=msk_label,
        #                      slice_index=depth // 2, save_path=save_path,)
        for slice_index in range(depth):
            save_path = os.path.join(fig_save_dir, f"gcam_slice_{slice_index}.png")

            show_slices_with_cam(volume=X, cam=cam_up, mask=msk_label,
                                 slice_index=slice_index, save_path=save_path,)

        print(f"Saved GCAM slices to: {fig_save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str,
                        default="/home/huangdn/Causal3D-Net/src/results/2025-11-05_02-24-16", required=False)
    parser.add_argument("--excel_path", type=str,
                        default="/home/huangdn/Causal3D-Net/src/dataset/dataset_for_gcam.xlsx", required=False)
    parser.add_argument("--cuda_id", type=int, default=5)
    args = parser.parse_args()

    run_gcam_visualization(
        model_dir=args.model_dir,
        excel_path=args.excel_path,
        cuda_id=args.cuda_id,
    )


