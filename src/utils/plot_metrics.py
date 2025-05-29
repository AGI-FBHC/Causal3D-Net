# -*- coding: utf-8 -*-
# @Time    : 2025/5/28 21:05
# @Author  : D.N. Huang
# @Email   : CarlCypress@yeah.net
# @FileName: plot_metrics.py
# @Project : Causal3D-Net
import os
import numpy as np
import matplotlib.pyplot as plt


def plot_combined_metrics(
    train_losses, test_losses,
    train_dices, test_dices,
    save_dir='metrics_figs', filename='metrics_curve.png'
):
    os.makedirs(save_dir, exist_ok=True)

    epochs = list(range(1, len(train_losses) + 1))

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # Loss subplot
    axs[0].plot(epochs, train_losses, label='Train Loss', color='blue')
    axs[0].plot(epochs, test_losses, label='Test Loss', color='red')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].set_title('Loss Curve')
    axs[0].legend()
    axs[0].grid(True)

    # Dice subplot
    axs[1].plot(epochs, train_dices, label='Train Dice', color='green')
    axs[1].plot(epochs, test_dices, label='Test Dice', color='orange')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Dice')
    axs[1].set_title('Dice Curve')
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename))
    plt.close()


if __name__ == '__main__':
    # 虚假数据生成
    num_epochs = 20
    train_losses = np.linspace(1.0, 0.2, num_epochs) + np.random.normal(0, 0.05, num_epochs)
    test_losses = np.linspace(1.2, 0.3, num_epochs) + np.random.normal(0, 0.07, num_epochs)
    train_dices = np.linspace(0.4, 0.9, num_epochs) + np.random.normal(0, 0.03, num_epochs)
    test_dices = np.linspace(0.35, 0.88, num_epochs) + np.random.normal(0, 0.04, num_epochs)

    # 保证 Dice 在 0~1 范围内
    train_dices = np.clip(train_dices, 0, 1)
    test_dices = np.clip(test_dices, 0, 1)

    # 调用绘图函数
    plot_combined_metrics(
        train_losses.tolist(), test_losses.tolist(),
        train_dices.tolist(), test_dices.tolist(),
        save_dir='/home/huangdn/Causal3D-Net/src/results/2025-05-28_12-55-27',
        filename='fake_metrics_curve.png'
    )
    pass
