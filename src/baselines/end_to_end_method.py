# -*- coding: utf-8 -*-
# @Time    : 2025/6/27 15:10
# @Author  : D.N. Huang
# @Email   : CarlCypress@yeah.net
# @FileName: end_to_end_method.py
# @Project : Causal3D-Net
import os
import logging
from tqdm import tqdm

import numpy as np
import pandas as pd

import torchio as tio

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.augmentation.window import Windowing
from src.augmentation.brightness import MultiplicativeBrightnessTransform
from src.augmentation.contrast import ContrastTransform
from src.augmentation.gamma import GammaTransform
from src.augmentation.gaussian_blur import GaussianBlurTransform
from src.augmentation.gaussian_noise import GaussianNoiseTransform
from src.augmentation.low_resolution import SimulateLowResolutionTransform
from src.dataset.PC_dataset import PCDataset
from src.training.train_baseline import train_one_epoch, test_one_epoch
from src.metric.compute_score import compute_multi_metrics, evaluate_test_result
from src.utils.plot_metrics import plot_training_metrics_for_baseline, plot_group_metrics


def ct_with_dl(train_excel, test_excel,
               cuda_id, model, current_dir,
               dimension=3, patch_size=384) -> dict:
    test = pd.read_excel(test_excel)
    test_result = {"center": test["center"],
                   "cancer": test["cancer"]}

    batch_size = 4
    initial_lr = 1e-2
    weight_decay = 3e-5
    num_epochs = 50
    device = torch.device(f"cuda:{cuda_id}" if torch.cuda.is_available() else "cpu")

    resize_shape = (50, 256, 256) if dimension == 3 else (50, patch_size, patch_size)  # for Hybrid
    pre_transform = tio.Compose([
        Windowing(window_center=70, window_width=340),
        tio.RescaleIntensity(out_min_max=(0, 1)),
        tio.Resize(resize_shape),
    ])
    aug_transform = tio.Compose([
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
        tio.RandomAffine(
            scales=(0.8, 1.2),
            degrees=10,
            isotropic=False,
            p=0.5
        ),
        tio.RandomElasticDeformation(
            num_control_points=11,
            max_displacement=2,
            locked_borders=2,
            p=0.3
        ),
        tio.RandomElasticDeformation(
            num_control_points=13,
            max_displacement=3,
            locked_borders=2,
            p=0.3
        )
    ])

    train_transform = tio.Compose([
        *pre_transform.transforms,  # unpack preprocessing
        *aug_transform.transforms  # unpack augmentation
    ])
    test_transform = pre_transform
    train_dataset = PCDataset(excel_path=train_excel,
                              transform=train_transform,
                              return_type=0,
                              dimension=dimension)
    test_dataset = PCDataset(excel_path=test_excel,
                             transform=test_transform,
                             return_type=0,
                             dimension=dimension)
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

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=initial_lr, weight_decay=weight_decay, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    model.to(device)
    y_pred, y_prob = None, None
    all_train_metrics = {
        'losses': [],
        'acc': [],
        'auc': [],
        'sensitivity': [],
        'specificity': [],
        'precision': [],
        'f1': [],
    }

    all_test_metrics = {
        'losses': [],
        'acc': [],
        'auc': [],
        'sensitivity': [],
        'specificity': [],
        'precision': [],
        'f1': [],
    }
    group_metric_keys = ["acc", "auc", "sensitivity", "specificity", "precision", "f1"]
    group_names = ["internal_test_1", "external_test_1", "external_test_2", "uncertainty_test"]
    group_metrics_by_epoch = {
        key: {group: [] for group in group_names} for key in group_metric_keys
    }
    for epoch in tqdm(range(num_epochs), desc="Training"):
        train_loss, train_pred, train_prob, train_target = train_one_epoch(model, train_loader, criterion, optimizer, device)
        scheduler.step()
        test_loss, test_pred, test_prob, test_target = test_one_epoch(model, test_loader, criterion, device)

        train_metrics = compute_multi_metrics(y_true=train_target, y_pred=train_pred, y_prob=train_prob)
        test_metrics = compute_multi_metrics(y_true=test_target, y_pred=test_pred, y_prob=test_prob)
        y_pred, y_prob = test_pred, test_prob

        test_result["y_pred"] = y_pred
        test_result["y_prob"] = y_prob
        all_train_metrics['losses'].append(train_loss)
        all_test_metrics['losses'].append(test_loss)
        for k in train_metrics:
            all_train_metrics[k].append(train_metrics[k])
            all_test_metrics[k].append(test_metrics[k])
        test_group_metrics = evaluate_test_result(test_result)
        for metric_key in group_metric_keys:
            for group_name in group_names:
                group_metrics_by_epoch[metric_key][group_name].append(
                    test_group_metrics[group_name][metric_key]
                )

        current_lr = optimizer.param_groups[0]['lr']
        log_msg = (f"[Epoch {epoch + 1}/{num_epochs}, LR {current_lr:.6f}]\n"
                   f"Train => Loss: {train_loss:.4f}, "
                   f"Accuracy: {train_metrics['acc']:.4f}, AUC: {train_metrics['auc']:.4f}, "
                   f"Sensitivity: {train_metrics['sensitivity']:.4f}, Specificity: {train_metrics['specificity']:.4f}, "
                   f"Precision: {train_metrics['precision']:.4f}, F1: {train_metrics['f1']:.4f}\n"
                   f"Test  => Loss: {test_loss:.4f}, "
                   f"Accuracy: {test_metrics['acc']:.4f}, AUC: {test_metrics['auc']:.4f}, "
                   f"Sensitivity: {test_metrics['sensitivity']:.4f}, Specificity: {test_metrics['specificity']:.4f}, "
                   f"Precision: {test_metrics['precision']:.4f}, F1: {test_metrics['f1']:.4f}\n")
        for group_name in group_names:
            group = test_group_metrics[group_name]
            log_msg += (f"Test({group_name}) => "
                        f"Accuracy: {group['acc']:.4f}, AUC: {group['auc']:.4f}, "
                        f"Sensitivity: {group['sensitivity']:.4f}, Specificity: {group['specificity']:.4f}, "
                        f"Precision: {group['precision']:.4f}, F1: {group['f1']:.4f}\n")
        logging.info(log_msg)

        plot_training_metrics_for_baseline(
            all_train_metrics,
            all_test_metrics,
            save_path=os.path.join(current_dir, "training_metrics.png"),
        )
        plot_group_metrics(
            group_accs=group_metrics_by_epoch['acc'],
            group_aucs=group_metrics_by_epoch['auc'],
            group_sensitivitys=group_metrics_by_epoch['sensitivity'],
            group_specificitys=group_metrics_by_epoch['specificity'],
            group_precisions=group_metrics_by_epoch['precision'],
            group_f1s=group_metrics_by_epoch['f1'],
            save_path=os.path.join(current_dir, "group_metrics.png")
        )

    return {
        "y_pred": y_pred,
        "y_prob": y_prob,
    }


if __name__ == '__main__':
    from src.models.Hybrid_Transformer.Hybrid.getmodel import get_model

    ps = 384
    m = get_model(
        num_classes=2,
        edge_size=ps,
        model_idx=f'Hybrid2_{ps}_401_test',
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        pretrained_backbone=False,  # 是否加载预训练CNN, 因channel数量改变, 故不可使用原加载预训练权重.
        use_cls_token=True,
        use_pos_embedding=True,
        use_att_module='SimAM'  # 使用 SimAM 注意力模块
    )
    ct_with_dl(train_excel="/home/huangdn/Causal3D-Net/src/dataset/dataset_for_train.xlsx",
               test_excel="/home/huangdn/Causal3D-Net/src/dataset/dataset_for_test.xlsx",
               cuda_id=4,
               model=m,
               current_dir="/home/huangdn/Causal3D-Net/src/results/debug",
               dimension=2,
               patch_size=ps)
    pass
