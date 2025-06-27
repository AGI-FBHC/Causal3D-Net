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
from src.models.VGG_2_5D import VGG25D
from src.training.train_baseline import train_one_epoch, test_one_epoch
from src.metric.compute_score import compute_multi_metrics
from src.utils.plot_metrics import plot_training_metrics_for_baseline


def ct_with_dl(train_excel, test_excel, cuda_id, model, current_dir) -> dict:
    """Simonyan, K. and Zisserman, A., 2014. Very deep convolutional networks for
    large-scale image recognition. arXiv preprint arXiv:1409.1556."""

    batch_size = 4
    initial_lr = 1e-2
    weight_decay = 3e-5
    num_epochs = 50
    device = torch.device(f"cuda:{cuda_id}" if torch.cuda.is_available() else "cpu")

    pre_transform = tio.Compose([
        Windowing(window_center=70, window_width=340),
        tio.RescaleIntensity(out_min_max=(0, 1)),
        tio.Resize((50, 256, 256)),
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
                              return_type=0)
    test_dataset = PCDataset(excel_path=test_excel,
                             transform=test_transform,
                             return_type=0)
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
    for epoch in tqdm(range(num_epochs), desc="Training"):
        train_loss, train_pred, train_prob, train_target = train_one_epoch(model, train_loader, criterion, optimizer, device)
        scheduler.step()
        test_loss, test_pred, test_prob, test_target = test_one_epoch(model, test_loader, criterion, device)

        train_metrics = compute_multi_metrics(y_true=train_target, y_pred=train_pred, y_prob=train_prob)
        test_metrics = compute_multi_metrics(y_true=test_target, y_pred=test_pred, y_prob=test_prob)
        y_pred, y_prob = test_pred, test_prob

        all_train_metrics['losses'].append(train_loss)
        all_test_metrics['losses'].append(test_loss)
        for k in train_metrics:
            all_train_metrics[k].append(train_metrics[k])
            all_test_metrics[k].append(test_metrics[k])

        plot_training_metrics_for_baseline(
            train_metrics,
            test_metrics,
            save_path=os.path.join(current_dir, "training_metrics.png"),
        )

    return {
        "y_pred": y_pred,
        "y_prob": y_prob,
    }


if __name__ == '__main__':

    pass