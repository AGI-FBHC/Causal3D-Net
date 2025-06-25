# -*- coding: utf-8 -*-
# @Time    : 2025/5/28 21:05
# @Author  : D.N. Huang
# @Email   : CarlCypress@yeah.net
# @FileName: plot_metrics.py
# @Project : Causal3D-Net
import os
import numpy as np
import matplotlib.pyplot as plt


def plot_loss_and_dice_metrics(
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


def plot_segmentation_and_classify_metrics(
        train_losses, test_losses,
        train_dices, test_dices,
        train_accs, test_accs,
        train_precs, test_precs,
        train_recalls, test_recalls,
        train_f1s, test_f1s,
        save_dir='metrics_figs', filename='all_metrics_curve.png'
):
    """
    绘制完整的训练指标曲线，包括：
    - 损失曲线
    - Dice分数曲线
    - 分类指标曲线（准确率、精确率、召回率、F1分数）

    参数:
        train_losses: 训练损失列表
        test_losses: 验证损失列表
        train_dices: 训练Dice分数列表
        test_dices: 验证Dice分数列表
        train_accs: 训练准确率列表
        test_accs: 验证准确率列表
        train_precs: 训练精确率列表
        test_precs: 验证精确率列表
        train_recalls: 训练召回率列表
        test_recalls: 验证召回率列表
        train_f1s: 训练F1分数列表
        test_f1s: 验证F1分数列表
        save_dir: 保存目录
        filename: 保存文件名
    """
    os.makedirs(save_dir, exist_ok=True)
    epochs = list(range(1, len(train_losses) + 1))

    # 创建更大的画布
    plt.figure(figsize=(18, 12))

    # 1. 损失曲线
    plt.subplot(2, 3, 1)
    plt.plot(epochs, train_losses, label='Train Loss', color='blue', linewidth=2)
    plt.plot(epochs, test_losses, label='Test Loss', color='red', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # 2. Dice分数曲线
    plt.subplot(2, 3, 2)
    plt.plot(epochs, train_dices, label='Train Dice', color='green', linewidth=2)
    plt.plot(epochs, test_dices, label='Test Dice', color='orange', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Dice Score')
    plt.title('Dice Score Curve')
    plt.ylim(0, 1)  # Dice分数范围0-1
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # 3. 准确率曲线
    plt.subplot(2, 3, 3)
    plt.plot(epochs, train_accs, label='Train Accuracy', color='purple', linewidth=2)
    plt.plot(epochs, test_accs, label='Test Accuracy', color='brown', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curve')
    plt.ylim(0, 1)  # 准确率范围0-1
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # 4. 精确率曲线
    plt.subplot(2, 3, 4)
    plt.plot(epochs, train_precs, label='Train Precision', color='cyan', linewidth=2)
    plt.plot(epochs, test_precs, label='Test Precision', color='magenta', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Precision')
    plt.title('Precision Curve')
    plt.ylim(0, 1)  # 精确率范围0-1
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # 5. 召回率曲线
    plt.subplot(2, 3, 5)
    plt.plot(epochs, train_recalls, label='Train Recall', color='teal', linewidth=2)
    plt.plot(epochs, test_recalls, label='Test Recall', color='gold', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Recall')
    plt.title('Recall Curve')
    plt.ylim(0, 1)  # 召回率范围0-1
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # 6. F1分数曲线（重点突出）
    plt.subplot(2, 3, 6)
    plt.plot(epochs, train_f1s, label='Train F1', color='darkblue', linewidth=3)
    plt.plot(epochs, test_f1s, label='Test F1', color='darkred', linewidth=3)
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title('F1 Score Curve (Primary Metric)', fontsize=14, fontweight='bold')
    plt.ylim(0, 1)  # F1分数范围0-1
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # 添加整体标题
    plt.suptitle('Training Metrics Overview', fontsize=16, fontweight='bold')

    # 调整布局
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # 为整体标题留出空间

    # 保存图像
    plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()


def plot_training_metrics(
    all_train_losses,
    all_train_cls_losses,
    all_test_cls_losses,
    all_train_accs,
    all_test_accs,
    all_train_precs,
    all_test_precs,
    all_train_recalls,
    all_test_recalls,
    all_train_aucs,
    all_test_aucs,
    save_path="training_metrics.png"
):
    epochs = list(range(1, len(all_train_losses) + 1))
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    axes = axes.flatten()

    # 1. All loss
    axes[0].plot(epochs, all_train_losses, label='Train Total Loss')
    axes[0].set_title('1. Train Total Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)

    # 2. Cls loss
    axes[1].plot(epochs, all_train_cls_losses, label='Train Cls Loss')
    axes[1].plot(epochs, all_test_cls_losses, label='Test Cls Loss')
    axes[1].set_title('2. Classification Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)

    # 3. Accuracy
    axes[2].plot(epochs, all_train_accs, label='Train Accuracy')
    axes[2].plot(epochs, all_test_accs, label='Test Accuracy')
    axes[2].set_title('3. Accuracy')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Accuracy')
    axes[2].legend()
    axes[2].grid(True)

    # 4. Recall
    axes[3].plot(epochs, all_train_recalls, label='Train Recall')
    axes[3].plot(epochs, all_test_recalls, label='Test Recall')
    axes[3].set_title('4. Recall')
    axes[3].set_xlabel('Epoch')
    axes[3].set_ylabel('Recall')
    axes[3].legend()
    axes[3].grid(True)

    # 5. Precision
    axes[4].plot(epochs, all_train_precs, label='Train Precision')
    axes[4].plot(epochs, all_test_precs, label='Test Precision')
    axes[4].set_title('5. Precision')
    axes[4].set_xlabel('Epoch')
    axes[4].set_ylabel('Precision')
    axes[4].legend()
    axes[4].grid(True)

    # 6. AUC
    axes[5].plot(epochs, all_train_aucs, label='Train AUC')
    axes[5].plot(epochs, all_test_aucs, label='Test AUC')
    axes[5].set_title('6. AUC')
    axes[5].set_xlabel('Epoch')
    axes[5].set_ylabel('AUC')
    axes[5].legend()
    axes[5].grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=400)
    plt.close()
    print(f"📈 Training metrics saved to: {save_path}")


def plot_group_metrics(
    group_accs, group_recalls, group_precisions, group_aucs,
    save_path="group_metrics.png"
):
    epochs = list(range(1, len(next(iter(group_accs.values()))) + 1))
    group_names = ['internal_test_1', 'external_test_1', 'external_test_2', 'uncertainty_test']
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    axs = axs.flatten()

    metric_dicts = [group_accs, group_recalls, group_precisions, group_aucs]
    metric_titles = ['Accuracy', 'Recall', 'Precision', 'AUC']

    for i, (metric_dict, title) in enumerate(zip(metric_dicts, metric_titles)):
        ax = axs[i]
        for group_name, color in zip(group_names, colors):
            values = metric_dict[group_name]
            ax.plot(epochs, values, label=group_name, color=color, linewidth=2)
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel(title, fontsize=12)
        ax.set_xlim([1, epochs[-1]])
        ax.grid(True)
        ax.legend()

    plt.tight_layout()
    plt.savefig(save_path)
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
    plot_loss_and_dice_metrics(
        train_losses.tolist(), test_losses.tolist(),
        train_dices.tolist(), test_dices.tolist(),
        save_dir='/home/huangdn/Causal3D-Net/src/results/2025-05-28_12-55-27',
        filename='fake_metrics_curve.png'
    )
    pass
