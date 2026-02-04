# -*- coding: utf-8 -*-
# @Time    : 2025/6/10 20:43
# @Author  : D.N. Huang
# @Email   : CarlCypress@yeah.net
# @FileName: train_Causal3DNet.py
# @Project : Causal3D-Net
import time
import logging
import os, argparse
import nibabel as nib
from tqdm import tqdm
from datetime import datetime
from collections import defaultdict

from typing import Union

import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import torchio as tio
import torch.nn as nn
import torch.cuda as cuda
import torch.optim as optim
from torch.utils.data import DataLoader

from sklearn.metrics import (accuracy_score,
                             roc_auc_score,
                             recall_score,
                             precision_score,
                             f1_score,
                             confusion_matrix)
from sklearn.model_selection import KFold, StratifiedKFold

from src.dataset.PC_dataset import PCDataset
from src.dataset.Seg_dataset import SegDataset
from src.models.Causal3DNet import SegNet, Causal3DNet
from src.augmentation.window import Windowing
from src.augmentation.brightness import MultiplicativeBrightnessTransform
from src.augmentation.contrast import ContrastTransform
from src.augmentation.gamma import GammaTransform
from src.augmentation.gaussian_blur import GaussianBlurTransform
from src.augmentation.gaussian_noise import GaussianNoiseTransform
from src.augmentation.low_resolution import SimulateLowResolutionTransform
from src.lr_scheduler.linear_lr import linear_lr_lambda
from src.utils.plot_metrics import (plot_loss_and_dice_metrics,
                                    plot_segmentation_and_classify_metrics,
                                    plot_training_metrics,
                                    plot_group_metrics)
from src.utils.init_weights import init_weights_kaiming, load_shared_weights
from src.utils.coefficient import compute_lambdas
from src.utils.data_type import to_number
from src.metric.loss import (DiceLoss,
                             MultiScaleSegmentationLoss,
                             MultiTaskLoss,
                             OrthogonalLoss,
                             SupervisedContrastiveLoss)
from src.metric.compute_score import compute_dice_score, specificity_score

import warnings

warnings.filterwarnings("ignore", category=UserWarning)


def train_seg(train_excel,
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
    num_epochs = 20
    device = torch.device(f"cuda:{cuda_id}" if torch.cuda.is_available() else "cpu")

    logging.info("Pancreas Structure Prior Module training")
    logging.info(f"train={train_excel} | test={test_excel} | "
                 f"bs={batch_size}, lr={initial_lr}, epochs={num_epochs}, device={device}")

    model = SegNet(mask_num=2)
    model.apply(init_weights_kaiming)
    model.to(device)

    criterion = MultiScaleSegmentationLoss()
    optimizer = optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=linear_lr_lambda(num_epochs)
    )

    patch_size = (40, 160, 256)
    half_patch = tuple(s // 2 for s in patch_size)
    margin = (8, 16, 16)
    pad_size = tuple(h + m for h, m in zip(half_patch, margin))

    pre_transform = tio.Compose([
        tio.ToCanonical(),
        Windowing(window_center=70, window_width=340),
        tio.RescaleIntensity(out_min_max=(0, 1)),
        tio.Resample((2.5, 1.0, 1.0)),
        tio.Pad(pad_size),
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
            num_control_points=7,
            max_displacement=3,
            locked_borders=2,
            p=0.3
        ),
        tio.RandomElasticDeformation(
            num_control_points=9,
            max_displacement=5,
            locked_borders=2,
            p=0.3
        )
    ])

    train_transform = tio.Compose([
        *pre_transform.transforms,  # unpack preprocessing
        *aug_transform.transforms,  # unpack augmentation
        tio.Pad(pad_size),
    ])
    test_transform = pre_transform

    train_dataset = SegDataset(excel_path=train_excel, transform=train_transform, verify_paths=True)
    test_dataset = SegDataset(excel_path=test_excel, transform=test_transform, verify_paths=True)

    sampler = tio.LabelSampler(patch_size=patch_size, label_name="mask")

    train_queue = tio.Queue(subjects_dataset=train_dataset,
                            max_length=1024,
                            samples_per_volume=8,
                            sampler=sampler,
                            num_workers=8,
                            shuffle_subjects=True,
                            shuffle_patches=True,)
    test_queue = tio.Queue(subjects_dataset=test_dataset,
                           max_length=256,
                           samples_per_volume=4,
                           sampler=sampler,
                           num_workers=2,
                           shuffle_subjects=False,
                           shuffle_patches=False,)

    train_loader = tio.SubjectsLoader(train_queue, batch_size=batch_size, num_workers=0, pin_memory=True)
    test_loader = tio.SubjectsLoader(test_queue, batch_size=batch_size, num_workers=0, pin_memory=True)

    all_train_losses, all_test_losses = [], []
    all_train_dices, all_test_dices = [], []

    best_dice = 0.0
    for epoch in tqdm(range(num_epochs)):
        model.train()
        train_loss, train_dice = 0.0, 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
            x = batch["image"][tio.DATA].to(device)
            y = batch["mask"][tio.DATA].to(device)
            optimizer.zero_grad()
            y_sgs, y_cls = model(x)
            loss = criterion(y_sgs, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            with torch.no_grad():
                highest_res = y_sgs[0]
                dice_score = compute_dice_score(highest_res, y)
                train_dice += dice_score.item()
        avg_train_loss = train_loss / len(train_loader)
        avg_train_dice = train_dice / len(train_loader)

        model.eval()
        test_loss, test_dice = 0.0, 0.0
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
                x = batch["image"][tio.DATA].to(device)
                y = batch["mask"][tio.DATA].to(device)

                y_sgs, y_cls = model(x)
                loss = criterion(y_sgs, y)
                test_loss += loss.item()

                dice_score = compute_dice_score(y_sgs[0], y)
                test_dice += dice_score.item()

        avg_test_loss = test_loss / len(test_loader)
        avg_test_dice = test_dice / len(test_loader)

        scheduler.step()

        all_train_losses.append(avg_train_loss)
        all_test_losses.append(avg_test_loss)
        all_train_dices.append(avg_train_dice)
        all_test_dices.append(avg_test_dice)

        current_lr = optimizer.param_groups[0]['lr']
        log_msg = (f"[Epoch {epoch + 1}/{num_epochs}, LR {current_lr:.6f}] "
                f"Train Loss: {avg_train_loss:.4f}, Train Dice: {avg_train_dice:.4f} | "
                f"Test Loss: {avg_test_loss:.4f}, Test Dice: {avg_test_dice:.4f}")
        logging.info(log_msg)

        if avg_test_dice > best_dice:
            best_dice = avg_test_dice
            torch.save(model.state_dict(), best_model_save_path)
            logging.info(f"Best model updated at epoch {epoch+1}, Dice: {best_dice:.4f}")

        torch.save(model.state_dict(), last_model_save_path)

        plot_loss_and_dice_metrics(
            all_train_losses, all_test_losses,
            all_train_dices, all_test_dices,
            save_dir=current_dir,
            filename='training_process.png'
        )


def train_Causal3DNet(train_excel: str, test_excel: str,
                      use_indi: int = 1, use_cent: int = 1,
                      orthogonal: int = 1,
                      adaptive: str = "reward_bad",
                      cuda_id: int =5,
                      output_dir: Union[str, None] = "/home/huangdn/Causal3D-Net/src/results",
                      model_weight: str = "/home/huangdn/Causal3D-Net/src/results/"
                                          "2025-06-21_06-49-30/last_model.pth",
                      logger: Union[logging.Logger, None] = None) -> dict:

    if logger is None:
        class NullLogger:
            def info(self, *args, **kwargs): pass
            def warning(self, *args, **kwargs): pass
            def error(self, *args, **kwargs): pass
        logger = NullLogger()

    current_dir = None
    diagnose_dir = None
    best_model_save_path = None
    last_model_save_path = None

    if output_dir:
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        current_dir = os.path.join(output_dir, current_time)
        diagnose_dir = os.path.join(current_dir, "diagnose")
        os.makedirs(diagnose_dir, exist_ok=True)

        log_filename = os.path.join(current_dir, "train.log")
        logging.basicConfig(
            filename=log_filename,
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            filemode='w'
        )
        logger = logging.getLogger()
        logger.addHandler(logging.StreamHandler())

        best_model_save_path = os.path.join(current_dir, "best_model.pth")
        last_model_save_path = os.path.join(current_dir, "last_model.pth")

    logger.info(f"Training dataset path is: {train_excel}")
    logger.info(f"Test dataset path is: {test_excel}")

    info_smg = (f"Using individual branch: {'Yes' if use_indi else 'No'}, "
                f"center branch: {'Yes' if use_cent else 'No'}, "
                f"orthogonal loss: {'Yes' if orthogonal else 'No'}, "
                f"adaptive mode: {adaptive}, in causal module.")
    logger.info(info_smg)

    batch_size = 8
    initial_lr = 1e-3
    weight_decay = 3e-5
    num_epochs = 50
    mid_1_epochs = 10
    mid_2_epochs = 20
    mid_1_transition_epochs = 5
    mid_2_transition_epochs = 5
    start_record_epoch = 10
    lambda_m, lambda_i, lambda_c = 1, 1, 1
    device = torch.device(f"cuda:{cuda_id}" if torch.cuda.is_available() else "cpu")

    center_groups = {
        'internal_test_1': [0, 3],
        'external_test_1': [6, 8],
        'external_test_2': [15, 16, 17],
        'uncertainty_test': [9, 10, 12],
    }

    model = Causal3DNet()
    model = load_shared_weights(model, model_weight)
    model.to(device)

    cls_criterion = nn.CrossEntropyLoss()
    suc_criterion = SupervisedContrastiveLoss()
    ort_criterion = OrthogonalLoss()

    optimizer = optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=linear_lr_lambda(num_epochs)
    )

    pre_transform = tio.Compose([
        Windowing(window_center=70, window_width=340),
        tio.RescaleIntensity(out_min_max=(0, 1)),
        tio.Resize((40, 160, 256)),
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
                              return_type=4)
    test_dataset = PCDataset(excel_path=test_excel,
                             transform=test_transform,
                             return_type=5)
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

    all_train_losses = []
    all_train_cls_losses, all_test_cls_losses = [], []
    all_train_accs, all_test_accs = [], []
    all_train_aucs, all_test_aucs = [], []
    all_train_sensitivitys, all_test_sensitivitys = [], []
    all_train_specificitys, all_test_specificitys = [], []
    all_train_precisions, all_test_precisions = [], []
    all_train_f1s, all_test_f1s = [], []

    group_accs = defaultdict(list)
    group_aucs = defaultdict(list)
    group_sensitivitys = defaultdict(list)
    group_specificitys = defaultdict(list)
    group_precisions = defaultdict(list)
    group_f1s = defaultdict(list)

    best_auc = .0
    for epoch in tqdm(range(num_epochs)):
        model.train()

        total_loss = 0
        total_cls_loss = 0
        train_preds = []
        train_probs = []
        train_targets = []

        for x, y_cls, _, center, cluster in train_loader:
            x, y_cls, center, cluster = x.to(device), y_cls.to(device), center.to(device), cluster.to(device)

            optimizer.zero_grad()

            ((y_indi, y_main, y_cent),
             (individual_confounder, classify_feature, center_confounder)) = model(x)

            l_c_main = cls_criterion(y_main * lambda_m, y_cls)
            # l_c_main = cls_criterion(y_main, y_cls)

            l_indi =  suc_criterion(individual_confounder, cluster) if use_indi else 0
            l_cent = suc_criterion(center_confounder, center) if use_cent else 0

            if epoch >= mid_1_epochs:
                alpha1 = min(1.0, (epoch - mid_1_epochs + 1) / mid_1_transition_epochs)
                l_o_im = ort_criterion(classify_feature, individual_confounder) \
                    if use_indi and orthogonal else 0
                l_o_cm = ort_criterion(classify_feature, center_confounder) \
                    if use_cent and orthogonal else 0

                if epoch >= mid_2_epochs:
                    alpha2 = min(1.0, (epoch - mid_2_epochs + 1) / mid_2_transition_epochs)
                    l_c_indi = cls_criterion(y_indi, y_cls) if use_indi else 0
                    l_c_cent = cls_criterion(y_cent, y_cls) if use_cent else 0

                    lambda_m, lambda_i, lambda_c = compute_lambdas(l_c_main.item(),
                                                                   to_number(l_indi + l_o_im + l_c_indi),
                                                                   to_number(l_cent + l_o_cm + l_c_cent),
                                                                   mode=adaptive)
                    loss = (lambda_m * l_c_main +
                            alpha1 * (lambda_i * (l_indi + l_o_im) + lambda_c * (l_cent + l_o_cm)) +
                            alpha2 * (lambda_i * l_c_indi + lambda_c * l_c_cent))
                else:
                    lambda_m, lambda_i, lambda_c = compute_lambdas(l_c_main.item(),
                                                                   to_number(l_indi + l_o_im),
                                                                   to_number(l_cent + l_o_cm),
                                                                   mode=adaptive)
                    loss = (lambda_m * l_c_main +
                            alpha1 * (lambda_i * (l_indi + l_o_im) + lambda_c * (l_cent + l_o_cm)))
            else:
                lambda_m, lambda_i, lambda_c = compute_lambdas(l_c_main.item(),
                                                               to_number(l_indi),
                                                               to_number(l_cent),
                                                               mode=adaptive)
                loss = lambda_m * l_c_main + lambda_i * l_indi + lambda_c * l_cent

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            total_cls_loss += l_c_main.item()

            # probs = torch.softmax(y_main * lambda_m, dim=1)[:, 1].detach().cpu().numpy()
            probs = torch.softmax(y_main, dim=1)[:, 1].detach().cpu().numpy()
            preds = (probs > 0.5).astype(int)
            targets = y_cls.detach().cpu().numpy()

            train_probs.extend(probs)
            train_preds.extend(preds)
            train_targets.extend(targets)

        scheduler.step()

        train_loss = total_loss / len(train_loader)
        train_cls_loss = total_cls_loss / len(train_loader)
        train_accuracy = accuracy_score(train_targets, train_preds)
        train_auc = roc_auc_score(train_targets, train_probs)
        train_sensitivity = recall_score(train_targets, train_preds, zero_division=0)
        train_specificity = specificity_score(train_targets, train_preds)
        train_precision = precision_score(train_targets, train_preds, zero_division=0)
        train_f1 = f1_score(train_targets, train_preds, zero_division=0)

        model.eval()

        group_metrics = defaultdict(dict)

        test_filenames = []
        test_probs = []
        test_preds = []
        test_targets = []
        test_centers = []

        total_test_cls_loss = 0

        with torch.no_grad():
            for filename, x, y_cls, _, center, _ in test_loader:
                x, y_cls = x.to(device), y_cls.to(device)

                ((_, y_main, _), _) = model(x)
                # l_c_main = cls_criterion(y_main, y_cls * lambda_m)
                l_c_main = cls_criterion(y_main, y_cls)
                total_test_cls_loss += l_c_main.item()

                # probs = torch.softmax(y_main * lambda_m, dim=1)[:, 1].cpu().numpy()
                probs = torch.softmax(y_main, dim=1)[:, 1].cpu().numpy()
                preds = (probs > 0.5).astype(int)
                targets = y_cls.cpu().numpy()

                test_filenames.extend(filename)
                test_probs.extend(probs)
                test_preds.extend(preds)
                test_targets.extend(targets)
                test_centers.extend(center)

        test_cls_loss = total_test_cls_loss / len(test_loader)
        test_accuracy = accuracy_score(test_targets, test_preds)
        test_auc = roc_auc_score(test_targets, test_probs)
        test_sensitivity = recall_score(test_targets, test_preds, zero_division=0)
        test_specificity = specificity_score(test_targets, test_preds)
        test_precision = precision_score(test_targets, test_preds, zero_division=0)
        test_f1 = f1_score(test_targets, test_preds, zero_division=0)

        all_train_losses.append(train_loss)
        all_train_cls_losses.append(train_cls_loss)

        all_train_accs.append(train_accuracy)
        all_train_aucs.append(train_auc)
        all_train_sensitivitys.append(train_sensitivity)
        all_train_specificitys.append(train_specificity)
        all_train_precisions.append(train_precision)
        all_train_f1s.append(train_f1)

        all_test_cls_losses.append(test_cls_loss)

        all_test_accs.append(test_accuracy)
        all_test_aucs.append(test_auc)
        all_test_sensitivitys.append(test_sensitivity)
        all_test_specificitys.append(test_specificity)
        all_test_precisions.append(test_precision)
        all_test_f1s.append(test_f1)

        test_preds = np.array(test_preds)
        test_probs = np.array(test_probs)
        test_targets = np.array(test_targets)
        test_centers = np.array(test_centers)

        for group_name, group_centers in center_groups.items():
            mask = np.isin(test_centers, group_centers)

            group_y_true = test_targets[mask]
            group_y_pred = test_preds[mask]
            group_y_prob = test_probs[mask]

            group_acc = accuracy_score(group_y_true, group_y_pred)
            group_auc = roc_auc_score(group_y_true, group_y_prob) if len(np.unique(group_y_true)) > 1 else 0.5
            group_sensitivity = recall_score(group_y_true, group_y_pred, zero_division=0)
            group_specificity = specificity_score(group_y_true, group_y_pred)
            group_precision = precision_score(group_y_true, group_y_pred, zero_division=0)
            group_f1 = f1_score(group_y_true, group_y_pred, zero_division=0)

            group_metrics[group_name] = {
                "acc": group_acc,
                "auc": group_auc,
                "sensitivity": group_sensitivity,
                "specificity": group_specificity,
                "precision": group_precision,
                "f1": group_f1,
            }

            group_accs[group_name].append(group_acc)
            group_aucs[group_name].append(group_auc)
            group_sensitivitys[group_name].append(group_sensitivity)
            group_specificitys[group_name].append(group_specificity)
            group_precisions[group_name].append(group_precision)
            group_f1s[group_name].append(group_f1)

        current_lr = optimizer.param_groups[0]['lr']

        log_msg = (f"[Epoch {epoch + 1}/{num_epochs}, LR {current_lr}]\n"
                   f"Train => Loss(all): {train_loss:.4f}, Loss(cls): {train_cls_loss:.4f}, "
                   f"Accuracy: {train_accuracy:.4f}, AUC: {train_auc:.4f}, "
                   f"Sensitivity: {train_sensitivity:.4f}, Specificity: {train_specificity:.4f}, "
                   f"Precision: {train_precision:.4f}, F1: {train_f1:.4f}\n"
                   f"Test(All)  => Loss(cls): {test_cls_loss:.4f}, "
                   f"Accuracy: {test_accuracy:.4f}, AUC: {test_auc:.4f}, "
                   f"Sensitivity: {test_sensitivity:.4f}, Specificity: {test_specificity:.4f}, "
                   f"Precision: {test_precision:.4f}, F1: {test_f1:.4f}\n")
        for group_name in ["internal_test_1", "external_test_1", "external_test_2", "uncertainty_test"]:
            metrics = group_metrics[group_name]
            log_msg += (f"Test({group_name}) => "
                        f"Accuracy: {metrics['acc']:.4f}, "
                        f"AUC: {metrics['auc']:.4f}, "
                        f"Sensitivity: {metrics['sensitivity']:.4f}, "
                        f"Specificity: {metrics['specificity']:.4f}, "
                        f"Precision: {metrics['precision']:.4f}, "
                        f"F1: {metrics['f1']:.4f}\n")
        logger.info(log_msg)

        if output_dir is not None:
            if test_auc > best_auc:
                best_auc = test_auc
                torch.save(model.state_dict(), best_model_save_path)
                logger.info(f"✅ Best model updated at epoch {epoch+1}, AUC: {best_auc:.4f}")

            if epoch >= start_record_epoch:
                test_result = pd.DataFrame({
                    "filename": test_filenames,
                    "center_id": test_centers,
                    "true_label": test_targets,
                    "predicted_prob": test_probs,
                    "predicted_label": test_preds,
                })
                test_result.to_csv(os.path.join(diagnose_dir, f"test_result_for_{epoch+1}.csv"), index=False)

            torch.save(model.state_dict(), last_model_save_path)

            plot_training_metrics(
                all_train_losses,
                all_train_cls_losses,
                all_test_cls_losses,
                all_train_accs,
                all_test_accs,
                all_train_aucs,
                all_test_aucs,
                all_train_sensitivitys,
                all_test_sensitivitys,
                all_train_specificitys,
                all_test_specificitys,
                all_train_precisions,
                all_test_precisions,
                all_train_f1s,
                all_test_f1s,
                save_path=os.path.join(current_dir, "training_metrics.png"),
            )

            plot_group_metrics(
                group_accs=group_accs,
                group_aucs=group_aucs,
                group_sensitivitys=group_sensitivitys,
                group_specificitys=group_specificitys,
                group_precisions=group_precisions,
                group_f1s=group_f1s,
                save_path=os.path.join(current_dir, "group_metrics.png")
            )

    final_metrics = {
        "accuracy": all_test_accs[-1],
        "auc": all_test_aucs[-1],
        "sensitivity": all_test_sensitivitys[-1],
        "specificity": all_test_specificitys[-1],
        "precision": all_test_precisions[-1],
        "f1": all_test_f1s[-1],
    }

    return final_metrics


def cross_validate_Causal3DNet(train_excel: str,
                               use_indi: int = 1,
                               use_cent: int = 1,
                               orthogonal: int = 1,
                               adaptive: str = "reward_bad",
                               folds: int = 10,
                               cuda_id: int = 5,
                               output_dir: Union[str, None] = "/home/huangdn/Causal3D-Net/src/results",
                               model_weight: str = "/home/huangdn/Causal3D-Net/src/results/"
                                                   "2025-06-21_06-49-30/last_model.pth") -> dict:
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    current_dir = os.path.join(output_dir, current_time)
    os.makedirs(current_dir, exist_ok=True)
    log_filename = os.path.join(current_dir, "cross_train.log")
    logging.basicConfig(
        filename=log_filename,
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger().addHandler(console)

    logging.info(f"CV dataset path is: {train_excel}")
    info_smg = (f"Using individual branch: {'Yes' if use_indi else 'No'}, "
                f"center branch: {'Yes' if use_cent else 'No'}, "
                f"orthogonal loss: {'Yes' if orthogonal else 'No'}, "
                f"adaptive mode: {adaptive}, in causal module.")
    logging.info(info_smg)

    data = pd.read_excel(train_excel)
    targets = data["cancer"].values

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)

    all_metrics = defaultdict(list)

    for fold, (train_idx, val_idx) in enumerate(skf.split(data, targets)):
        logging.info(f"\n🚀 Fold {fold + 1}/{folds}")
        train_fold = data.iloc[train_idx].reset_index(drop=True)
        val_fold = data.iloc[val_idx].reset_index(drop=True)

        train_path = os.path.join(current_dir, f"cv_train_fold_{fold + 1}.xlsx")
        val_path = os.path.join(current_dir, f"cv_val_fold_{fold + 1}.xlsx")
        train_fold.to_excel(train_path, index=False)
        val_fold.to_excel(val_path, index=False)

        metrics = train_Causal3DNet(
            train_excel=train_path,
            test_excel=val_path,
            use_indi=use_indi,
            use_cent=use_cent,
            orthogonal=orthogonal,
            adaptive=adaptive,
            cuda_id=cuda_id,
            output_dir=None,
            model_weight=model_weight,
            logger=logging.getLogger()
        )

        for key, value in metrics.items():
            all_metrics[key].append(value)

    summary = {}
    for key, values in all_metrics.items():
        mean = np.mean(values)
        std = np.std(values)
        summary[key] = {
            "folds": values,
            "mean": mean,
            "std": std
        }

    # Save to CSV
    rows = []
    for i in range(folds):
        row = {"fold": i + 1}
        for metric in summary:
            row[metric] = summary[metric]["folds"][i]
        rows.append(row)

    mean_row = {"fold": "mean"}
    std_row = {"fold": "std"}
    for metric in summary:
        mean_row[metric] = summary[metric]["mean"]
        std_row[metric] = summary[metric]["std"]
    rows.append(mean_row)
    rows.append(std_row)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(current_dir, "cross_validation_results.csv"), index=False)
    logging.info("\n✅ Cross-validation results saved to 'cross_validation_results.csv'")

    return summary


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Segmentation training")
    parser.add_argument("--train", type=str,
                        default="/home/huangdn/Causal3D-Net/src/dataset/dataset_for_train.xlsx",
                        # required=True,
                        help="path to training dataset")
    parser.add_argument("--test", type=str,
                        default="/home/huangdn/Causal3D-Net/src/dataset/dataset_for_test.xlsx",
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
    train_seg(
        train_excel=args.train,
        test_excel=args.test,
        cuda_id=args.cuda,
        output_dir=args.outdir
    )

    # parser = argparse.ArgumentParser(description="Causal 3D Net training")
    # parser.add_argument("--train", type=str,
    #                     default="/home/huangdn/Causal3D-Net/src/dataset/dataset_for_train.xlsx",
    #                     # required=True,
    #                     help="path to training dataset")
    # parser.add_argument("--test", type=str,
    #                     default="/home/huangdn/Causal3D-Net/src/dataset/dataset_for_test.xlsx",
    #                     # required=True,
    #                     help="path to testing dataset")
    # parser.add_argument("--indi", type=int,
    #                     default=1,
    #                     choices=[0, 1],
    #                     help="whether to use the individual branch in causal methods")
    # parser.add_argument("--cent", type=int,
    #                     default=1,
    #                     choices=[0, 1],
    #                     help="whether to use the center branch in causal methods")
    # parser.add_argument("--orth", type=float,
    #                     default=1,
    #                     choices=[0, 1],
    #                     help="whether to use the orthogonal loss in causal methods")
    # parser.add_argument("--adapt", type=str,
    #                     default="none",
    #                     choices=["none", "reward_bad", "reward_good"],
    #                     help="adaptive loss method")
    # parser.add_argument("--cuda", type=int,
    #                     default=5,
    #                     required=False,
    #                     help="index of GPU to use")
    # parser.add_argument("--outdir", type=str,
    #                     default="/home/huangdn/Causal3D-Net/src/results",
    #                     required=False,
    #                     help="output directory")
    # parser.add_argument("--weight", type=str,
    #                     default="/home/huangdn/Causal3D-Net/src/results/"
    #                             "2025-06-21_06-49-30/best_model.pth",
    #                     required=False,
    #                     help="segmentation trained model weight")
    # args = parser.parse_args()
    # train_Causal3DNet(
    #     train_excel=args.train,
    #     test_excel=args.test,
    #     use_indi=args.indi,
    #     use_cent=args.cent,
    #     orthogonal=args.orth,
    #     adaptive=args.adapt,
    #     cuda_id=args.cuda,
    #     output_dir=args.outdir,
    #     model_weight=args.weight,
    # )
    #
    # parser = argparse.ArgumentParser(description="Causal 3D Net cross-validation training")
    # parser.add_argument("--train", type=str,
    #                     default="/home/huangdn/Causal3D-Net/src/dataset/dataset_for_train.xlsx",
    #                     required=True,
    #                     help="path to training dataset (Excel file)")
    # parser.add_argument("--indi", type=int,
    #                     default=1,
    #                     choices=[0, 1],
    #                     help="whether to use the individual branch in causal methods")
    # parser.add_argument("--cent", type=int,
    #                     default=1,
    #                     choices=[0, 1],
    #                     help="whether to use the center branch in causal methods")
    # parser.add_argument("--orth", type=float,
    #                     default=1,
    #                     choices=[0, 1],
    #                     help="whether to use the orthogonal loss in causal methods")
    # parser.add_argument("--adapt", type=str,
    #                     default="none",
    #                     choices=["none", "reward_bad", "reward_good"],
    #                     help="adaptive loss method")
    # parser.add_argument("--folds", type=int,
    #                     default=10,
    #                     help="number of cross-validation folds")
    # parser.add_argument("--cuda", type=int,
    #                     default=5,
    #                     help="index of GPU to use")
    # parser.add_argument("--outdir", type=str,
    #                     default="/home/huangdn/Causal3D-Net/src/results",
    #                     help="output directory for results and logs")
    # parser.add_argument("--weight", type=str,
    #                     default="/home/huangdn/Causal3D-Net/src/results/"
    #                             "2025-06-21_06-49-30/best_model.pth",
    #                     required=False,
    #                     help="segmentation trained model weight")
    # args = parser.parse_args()
    #
    # summary = cross_validate_Causal3DNet(
    #     train_excel=args.train,
    #     use_indi=args.indi,
    #     use_cent=args.cent,
    #     orthogonal=args.orth,
    #     adaptive=args.adapt,
    #     output_dir=args.outdir,
    #     folds=args.folds,
    #     cuda_id=args.cuda,
    #     model_weight=args.weight,
    # )
    # pass
