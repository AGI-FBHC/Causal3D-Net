# -*- coding: utf-8 -*-
# @Time    : 2025/8/22 17:01
# @Author  : D.N. Huang
# @Email   : CarlCypress@yeah.net
# @FileName: visualization_of_experimental_results.py
# @Project : Causal3D-Net
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def vis_10_folds_cv(
        csv_path="/home/huangdn/Causal3D-Net/src/results/2025-07-17_14-09-09/cross_validation_results.csv",
        save_path="/home/huangdn/Causal3D-Net/src/logging_record/10_folds_cross_validation_results.png"
):
    # 设置字体
    # plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 15

    # 读取 CSV
    csv = pd.read_csv(csv_path)
    df = csv[csv['fold'].apply(lambda x: str(x).isdigit())].copy()

    # 指标
    metrics = ['accuracy', 'auc', 'sensitivity', 'specificity', 'precision', 'f1']
    metric_labels = ['ACC', 'AUC', 'SEN', 'SPE', 'PREC', 'F1']
    df_long = df.melt(id_vars=['fold'], value_vars=metrics, var_name='Metric', value_name='Value')

    # 可视化风格
    sns.set_style("white")
    plt.figure(figsize=(10, 6))

    # 调色板
    palette = sns.color_palette("Set2", n_colors=len(metrics))

    # 小提琴图，不显示内部线
    sns.violinplot(x='Metric', y='Value', data=df_long, palette=palette, inner=None)

    # 叠加箱线图，控制宽度
    sns.boxplot(x='Metric', y='Value', data=df_long, width=0.2,
                showcaps=True,
                boxprops={'facecolor':'white','edgecolor':'black'},
                showfliers=False,   # 不显示离群点
                whiskerprops={'linewidth':1.5},
                saturation=1)

    # 坐标轴设置
    plt.xlabel('')
    plt.ylabel('')
    plt.ylim(0.7, 1.05)
    plt.grid(False)
    plt.tick_params(axis='y', which='both', left=True, right=False, length=3)
    plt.tick_params(axis='x', which='both', bottom=True, top=False, length=3)

    # x轴刻度名称
    plt.xticks(ticks=range(len(metrics)), labels=metric_labels)

    # 保存图片
    plt.savefig(save_path, dpi=1000, bbox_inches='tight', transparent=False)
    plt.show()



def vis_ablation(excel_path="/home/huangdn/Causal3D-Net/src/logging_record/ablation.xlsx",
                 save_path="/home/huangdn/Causal3D-Net/src/logging_record/model_ablation.png"):

    plt.rcParams['font.size'] = 15

    order = [
        "baseline",
        "indi",
        "indi+ot",
        "cent",
        "cent+ot",
        "indi+cent",
        "indi+cent+ot"
    ]

    excel = pd.read_excel(excel_path).iloc[:, :-1]
    excel = excel[["name", "dataset", "Accuracy"]]

    datasets = excel["dataset"].unique()
    n_dataset = len(datasets)
    n_group = len(order)

    # 控制参数
    bar_width = 0.08     # 每根柱子的宽度
    inner_gap = 0.02     # 组内柱子之间的间隔
    group_gap = 0.2      # 组与组之间的间隔

    # 每个 dataset 在 x 轴上的起始位置
    group_width = n_group * bar_width + (n_group - 1) * inner_gap
    x_positions = np.arange(n_dataset) * (group_width + group_gap)

    palette = sns.color_palette("Set2", n_group)

    plt.figure(figsize=(10, 6))

    for i, name in enumerate(order):
        data_i = excel[excel["name"] == name].set_index("dataset").loc[datasets]["Accuracy"].values
        bar_x = x_positions + i * (bar_width + inner_gap)
        plt.bar(bar_x,
                data_i,
                width=bar_width,
                label=name,
                color=palette[i],
                edgecolor="black",
                linewidth=0.5
                )

    # 设置x轴刻度在每组的中心
    plt.xticks(x_positions + group_width/2 - bar_width/2, datasets)

    plt.ylabel("ACC")
    plt.ylim(0.65, 1)
    plt.legend(
        title="",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),  # 横向居中，纵向往下
        ncol=4,  # 图例分成多列，避免太宽
        frameon=False  # 去掉外框（可选）
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=600, bbox_inches='tight', transparent=False)
    plt.show()


if __name__ == '__main__':
    # vis_10_folds_cv()
    vis_ablation()
    pass



