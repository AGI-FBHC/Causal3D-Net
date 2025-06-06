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

from sklearn.metrics import (accuracy_score,
                             precision_score,
                             recall_score,
                             f1_score)

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
from src.lr_scheduler.linear_lr import linear_lr_lambda
from src.utils.plot_metrics import (plot_loss_and_dice_metrics,
                                    plot_segmentation_and_classify_metrics)
from src.utils.init_weights import init_weights_kaiming, load_shared_weights
from src.metric.loss import (DiceLoss,
                             MultiScaleSegmentationLoss,
                             compute_dice_score,
                             MultiTaskLoss)

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
    num_epochs = 200
    device = torch.device(f"cuda:{cuda_id}" if torch.cuda.is_available() else "cpu")

    model = SegNet(mask_num=2)
    model.apply(init_weights_kaiming)
    model.to(device)

    criterion = MultiScaleSegmentationLoss()
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
        # tio.RandomAffine(
        #     scales=(0.8, 1.2),
        #     degrees=10,
        #     isotropic=False,
        #     p=0.5),
        # tio.RandomElasticDeformation(
        #     num_control_points=7,
        #     max_displacement=3,
        #     locked_borders=2,
        #     p=0.3)
    ])

    train_transform = tio.Compose([
        *pre_transform.transforms,  # unpack preprocessing
        *aug_transform.transforms  # unpack augmentation
    ])
    test_transform = pre_transform

    train_dataset = PCDataset(excel_path=train_excel,
                              transform=train_transform,
                              return_type=1)
    test_dataset = PCDataset(excel_path=test_excel,
                             transform=test_transform,
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
                highest_res = y_sgs[0]
                dice_score = compute_dice_score(highest_res, y)
                train_dice += dice_score.item()
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


def stage_2_train(train_excel,
                  test_excel,
                  cuda_id=5,
                  output_dir: str = "/home/huangdn/Causal3D-Net/src/results",
                  model_weight: str = "/home/huangdn/Causal3D-Net/src/results/"
                                      "2025-06-01_05-55-00/best_model.pth"):
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
    initial_lr_for_sgs = 1e-4
    initial_lr_for_cls = 1e-3
    weight_decay = 3e-5
    num_epochs = 50
    device = torch.device(f"cuda:{cuda_id}" if torch.cuda.is_available() else "cpu")

    model = MultiTask3DCNN(mask_num=2, cls_num=2)
    model = load_shared_weights(model, weight_path=model_weight)
    model.to(device)
    logging.info(f"✅ Stage 1 model weights {model_weight} loaded successfully.")

    cls_decoder_params = []
    other_params = []

    for name, param in model.named_parameters():
        if 'cls_decoder' in name:  # 分类头参数
            cls_decoder_params.append(param)
            logging.info(f"分类头参数: {name} (LR={initial_lr_for_cls:.0e})")
        else:  # 其他参数（共享部分）
            other_params.append(param)
            logging.info(f"共享参数: {name} (LR={initial_lr_for_sgs:.0e})")

    cls_param_count = sum(p.numel() for p in cls_decoder_params)
    shared_param_count = sum(p.numel() for p in other_params)
    logging.info(f"分类头参数数量: {cls_param_count}, 使用学习率: {initial_lr_for_cls:.0e}")
    logging.info(f"共享参数数量: {shared_param_count}, 使用学习率: {initial_lr_for_sgs:.0e}")

    param_groups = [
        {'params': other_params, 'lr': initial_lr_for_sgs},
        {'params': cls_decoder_params, 'lr': initial_lr_for_cls}
    ]

    # criterion = MultiTaskLoss()
    cls_criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(param_groups, weight_decay=weight_decay)
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
        # tio.RandomAffine(
        #     scales=(0.8, 1.2),
        #     degrees=10,
        #     isotropic=False,
        #     p=0.5),
        # tio.RandomElasticDeformation(
        #     num_control_points=7,
        #     max_displacement=3,
        #     locked_borders=2,
        #     p=0.3)
    ])

    train_transform = tio.Compose([
        *pre_transform.transforms,  # unpack preprocessing
        *aug_transform.transforms  # unpack augmentation
    ])
    test_transform = pre_transform

    train_dataset = PCDataset(excel_path=train_excel,
                              transform=train_transform,
                              return_type=2)
    test_dataset = PCDataset(excel_path=test_excel,
                             transform=test_transform,
                             return_type=2)
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
    all_train_accs, all_test_accs = [], []
    all_train_precs, all_test_precs = [], []
    all_train_recalls, all_test_recalls = [], []
    all_train_f1s, all_test_f1s = [], []

    best_f1 = 0.0
    for epoch in tqdm(range(num_epochs)):
        model.train()
        train_loss, train_dice = 0.0, 0.0
        all_cls_preds, all_cls_labels = [], []

        for x, y_sgs, y_cls in train_loader:
            x, y_sgs, y_cls = x.to(device), y_sgs.to(device), y_cls.to(device)
            optimizer.zero_grad()

            y_hat_sgs, y_hat_cls = model(x)
            # loss, _, _ = criterion(y_hat_sgs, y_hat_cls, y_sgs, y_cls)
            cls_loss = cls_criterion(y_hat_cls, y_cls)
            # loss.backward()
            cls_loss.backward()
            optimizer.step()
            # train_loss += loss.item()
            train_loss += cls_loss.item()

            with torch.no_grad():
                dice_score = compute_dice_score(y_hat_sgs[0], y_sgs)
                train_dice += dice_score.item()
                preds = torch.argmax(y_hat_cls, dim=1)

                all_cls_preds.extend(preds.cpu().numpy())
                all_cls_labels.extend(y_cls.cpu().numpy())

        train_acc = accuracy_score(all_cls_labels, all_cls_preds)
        train_prec = precision_score(all_cls_labels,
                                     all_cls_preds,
                                     average='macro',
                                     zero_division=0)
        train_recall = recall_score(all_cls_labels,
                                    all_cls_preds,
                                    average='macro',
                                    zero_division=0)
        train_f1 = f1_score(all_cls_labels,
                            all_cls_preds,
                            average='macro',
                            zero_division=0)

        avg_train_loss = train_loss / len(train_loader)
        avg_train_dice = train_dice / len(train_loader)

        model.eval()
        test_loss, test_dice = 0.0, 0.0
        test_cls_preds, test_cls_labels = [], []

        with torch.no_grad():
            for x, y_sgs, y_cls in test_loader:
                x, y_sgs, y_cls = x.to(device), y_sgs.to(device), y_cls.to(device)

                y_hat_sgs, y_hat_cls = model(x)
                # loss, _, _ = criterion(y_hat_sgs, y_hat_cls, y_sgs, y_cls)
                loss = cls_criterion(y_hat_cls, y_cls)
                test_loss += loss.item()

                dice_score = compute_dice_score(y_hat_sgs[0], y_sgs)
                test_dice += dice_score.item()

                preds = torch.argmax(y_hat_cls, dim=1)
                test_cls_preds.extend(preds.cpu().numpy())
                test_cls_labels.extend(y_cls.cpu().numpy())

        test_acc = accuracy_score(test_cls_labels, test_cls_preds)
        test_prec = precision_score(test_cls_labels, test_cls_preds, average='macro', zero_division=0)
        test_recall = recall_score(test_cls_labels, test_cls_preds, average='macro', zero_division=0)
        test_f1 = f1_score(test_cls_labels, test_cls_preds, average='macro', zero_division=0)

        avg_test_loss = test_loss / len(test_loader)
        avg_test_dice = test_dice / len(test_loader)

        scheduler.step()

        all_train_losses.append(avg_train_loss)
        all_train_dices.append(avg_train_dice)
        all_train_accs.append(train_acc)
        all_train_precs.append(train_prec)
        all_train_recalls.append(train_recall)
        all_train_f1s.append(train_f1)
        all_test_losses.append(avg_test_loss)
        all_test_dices.append(avg_test_dice)
        all_test_accs.append(test_acc)
        all_test_precs.append(test_prec)
        all_test_recalls.append(test_recall)
        all_test_f1s.append(test_f1)

        current_lr_shared = optimizer.param_groups[0]['lr']
        current_lr_cls = optimizer.param_groups[1]['lr']

        log_msg = (f"[Epoch {epoch + 1}/{num_epochs}, "
                   f"Shared LR {current_lr_shared:.2e}, Cls LR {current_lr_cls:.2e}]\n"
                   f"Train => Loss: {avg_train_loss:.4f}, Dice: {avg_train_dice:.4f}, "
                   f"Acc: {train_acc:.4f}, Prec: {train_prec:.4f}, "
                   f"Recall: {train_recall:.4f}, F1: {train_f1:.4f}\n"
                   f"Test  => Loss: {avg_test_loss:.4f}, Dice: {avg_test_dice:.4f}, "
                   f"Acc: {test_acc:.4f}, Prec: {test_prec:.4f}, "
                   f"Recall: {test_recall:.4f}, F1: {test_f1:.4f}")
        logging.info(log_msg)

        if test_f1 > best_f1:
            best_f1 = test_f1
            torch.save(model.state_dict(), best_model_save_path)
            logging.info(f"Best model updated at epoch {epoch+1}, F1: {best_f1:.4f}")

        torch.save(model.state_dict(), last_model_save_path)

        plot_segmentation_and_classify_metrics(
            all_train_losses, all_test_losses,
            all_train_dices, all_test_dices,
            all_train_accs, all_test_accs,
            all_train_precs, all_test_precs,
            all_train_recalls, all_test_recalls,
            all_train_f1s, all_test_f1s,
            save_dir=current_dir,
            filename=f'training_process.png'
        )
    pass


if __name__ == '__main__':
    ######################### stage 1 #########################
    # parser = argparse.ArgumentParser(description="PANDA model stage 1 training")
    # parser.add_argument("--train", type=str,
    #                     default="/home/huangdn/Causal3D-Net/src/dataset/train_dataset.xlsx",
    #                     # required=True,
    #                     help="path to training dataset")
    # parser.add_argument("--test", type=str,
    #                     default="/home/huangdn/Causal3D-Net/src/dataset/test_dataset.xlsx",
    #                     # required=True,
    #                     help="path to testing dataset")
    # parser.add_argument("--cuda", type=int,
    #                     default=5,
    #                     required=False,
    #                     help="index of GPU to use")
    # parser.add_argument("--outdir", type=str,
    #                     default="/home/huangdn/Causal3D-Net/src/results",
    #                     required=False,
    #                     help="output directory")
    # args = parser.parse_args()
    # stage_1_train(
    #     train_excel=args.train,
    #     test_excel=args.test,
    #     cuda_id=args.cuda,
    #     output_dir=args.outdir,
    # )
    ######################### stage 1 #########################

    ######################### stage 2 #########################
    parser = argparse.ArgumentParser(description="PANDA model stage 2 training")
    parser.add_argument("--train", type=str,
                        default="/home/huangdn/Causal3D-Net/src/dataset/train_dataset.xlsx",
                        # required=True,
                        help="path to training dataset")
    parser.add_argument("--test", type=str,
                        default="/home/huangdn/Causal3D-Net/src/dataset/test_dataset.xlsx",
                        # required=True,
                        help="path to testing dataset")
    parser.add_argument("--cuda", type=int,
                        default=1,
                        required=False,
                        help="index of GPU to use")
    parser.add_argument("--outdir", type=str,
                        default="/home/huangdn/Causal3D-Net/src/results",
                        required=False,
                        help="output directory")
    parser.add_argument("--weight", type=str,
                        default="/home/huangdn/Causal3D-Net/src/results/"
                                "2025-06-01_05-55-00/best_model.pth",
                        required=False,
                        help="stage 1 trained model weight")
    args = parser.parse_args()
    stage_2_train(
        train_excel=args.train,
        test_excel=args.test,
        cuda_id=args.cuda,
        output_dir=args.outdir,
        model_weight=args.weight,
    )
    ######################### stage 2 #########################
    pass
