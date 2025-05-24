# -*- coding: utf-8 -*-
# @Time    : 2025/4/26 20:37
# @Author  : D.N. Huang
# @Email   : CarlCypress@yeah.net
# @FileName: train.py
# @Project : Causal3D-Net
import torch
import torchio as tio
import torch.nn as nn
import torch.cuda as cuda
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
import logging
import os, argparse
import nibabel as nib
from tqdm import tqdm
from src.dataset.PC_dataset import *
from src.models.ResNet import generate_model
from src.models.ViT import ViTClassifier
from src.models.PANDA import SegNet, MultiTask3DCNN
from src.utils.window import *
import matplotlib.pyplot as plt
from datetime import datetime

import warnings

warnings.filterwarnings("ignore", message=".*Using TorchIO images without a torchio.SubjectsLoader.*")


def training(train_excel, test_excel, output_dir, logging_dir):
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = os.path.join(logging_dir, f"{current_time}.log")
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
    learning_rate = 0.001
    num_epochs = 100
    device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")

    # model = generate_model(10, n_input_channels=1, n_classes=2).to(device)  # 3D ResNet
    model = ViTClassifier(img_size=(128, 256, 256), num_classes=2).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    transform = tio.Compose([
        Windowing(window_center=70, window_width=340),
        tio.RescaleIntensity(out_min_max=(0, 1)),
        tio.Resize((128, 256, 256)),
    ])
    train_dataset = PCDataset(excel_path=train_excel, transform=transform)
    test_dataset = PCDataset(excel_path=test_excel, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    best_val_loss = float('inf')
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in tqdm(range(num_epochs)):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for _, X, y in train_loader:
            # return None
            X, y = X.to(device, dtype=torch.float), y.to(device, dtype=torch.long)
            optimizer.zero_grad()
            # y_hat = model(X)  # for 3D-ResNet
            y_hat, _ = model(X)  # for ViT
            loss = loss_fn(y_hat, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * X.size(0)
            train_correct += (y_hat.argmax(1) == y).sum().item()
            train_total += y.size(0)

        avg_train_loss = train_loss / train_total
        train_acc = train_correct / train_total
        train_losses.append(avg_train_loss)
        train_accs.append(train_acc)

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for _, X, y in test_loader:
                X, y = X.to(device, dtype=torch.float), y.to(device, dtype=torch.long)
                # y_hat = model(X)  # for 3D-ResNet
                y_hat, _ = model(X)
                loss = loss_fn(y_hat, y)

                val_loss += loss.item() * X.size(0)
                val_correct += (y_hat.argmax(1) == y).sum().item()
                val_total += y.size(0)

        avg_val_loss = val_loss / val_total
        val_acc = val_correct / val_total
        val_losses.append(avg_val_loss)
        val_accs.append(val_acc)

        # Logging
        log_msg = (f"\n📘 Epoch {epoch+1}: "
                   f"Train Loss={avg_train_loss:.4f}, Acc={train_acc:.4f} | "
                   f"Val Loss={avg_val_loss:.4f}, Acc={val_acc:.4f}")
        print(log_msg)
        logging.info(log_msg)

        # 保存最优模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pth'))
            best_msg = f"✅ Saved best model at epoch {epoch} with val loss {best_val_loss:.4f}"
            print(best_msg)
            logging.info(best_msg)

        # 实时更新训练曲线图
        plt.figure(figsize=(10, 5))
        plt.clf()

        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss Curve')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(train_accs, label='Train Acc')
        plt.plot(val_accs, label='Val Acc')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Accuracy Curve')
        plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(logging_dir, f"{current_time}.png"))
        plt.close()

    torch.save(model.state_dict(), os.path.join(output_dir, 'final_model.pth'))
    print("✅ Training complete. Final model saved.")
    logging.info("✅ Training complete. Final model saved.")


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Causal 3D Net model training")
    parser.add_argument(
        "--train", type=str,
        default="/home/huangdn/Causal3D-Net/src/dataset/train_dataset.xlsx",
        # required=True,
        help="Excel path for model training image set."
    )
    parser.add_argument(
        "--test", type=str,
        default="/home/huangdn/Causal3D-Net/src/dataset/test_dataset.xlsx",
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
        "--log_dir", type=str,
        default="/home/huangdn/Causal3D-Net/src/logging_record",
        help="Logging record dir path."
    )
    args = parser.parse_args()
    training(
        args.train,
        args.test,
        args.outdir,
        args.log_dir
    )

