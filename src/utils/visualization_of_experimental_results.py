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


def draw_2dim_scatter(X_2d, y, title, xlabel, ylabel, save_path_png, is_center_group=False):
    plt.figure(figsize=(7, 6))

    if is_center_group:
        unique_groups = ['test I', 'test II', 'test III']
        cmap = plt.cm.get_cmap("tab10", len(unique_groups))
        color_map = {group: cmap(i) for i, group in enumerate(unique_groups)}

        for g in unique_groups:
            mask = (y == g)
            plt.scatter(
                X_2d[mask, 0], X_2d[mask, 1],
                color=color_map[g],
                s=15, alpha=0.8,
                label=g
            )
        plt.legend(title="Center Group")

    else:
        unique_labels = sorted(np.unique(y))
        cmap = plt.cm.get_cmap("tab10", len(unique_labels))

        for idx, lab in enumerate(unique_labels):
            mask = (y == lab)
            plt.scatter(
                X_2d[mask, 0], X_2d[mask, 1],
                color=cmap(idx),
                s=15, alpha=0.8,
                label=f"{lab}"
            )
        plt.legend(title="Label")

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(save_path_png)
    print(f"Saved PNG: {save_path_png}")
    plt.close()


def vis_10_folds_cv(
        csv_path="/home/huangdn/Causal3D-Net/src/logging_record/cross_validation_results.csv",
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


def vis_CCLM(excel_path="/home/huangdn/Causal3D-Net/src/logging_record/CCLM.xlsx",
             save_path="/home/huangdn/Causal3D-Net/src/logging_record/CCLM.svg"):

    # ===== 图形风格设置（论文推荐）=====
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 20
    plt.rcParams['axes.labelsize'] = 20
    plt.rcParams['xtick.labelsize'] = 20
    plt.rcParams['ytick.labelsize'] = 20
    plt.rcParams['legend.fontsize'] = 20
    plt.rcParams['legend.title_fontsize'] = 20
    plt.rcParams['svg.fonttype'] = 'none'
    plt.rcParams['axes.linewidth'] = 1

    # ===== 模型顺序 =====
    order = [
        "Baseline",
        "Baseline+ECRM",
        "Baseline+ICRM",
        "CCLM(ECRM+ICRM)"
    ]

    # ===== 指标 =====
    metrics = ["Acc", "AUC", "Sen", "Spe", "Prec", "F1"]

    df = pd.read_excel(excel_path)
    df = df[df["Model"].isin(order)]

    n_model = len(order)
    n_metric = len(metrics)

    # ===== 柱状图参数 =====
    bar_width = 0.12
    inner_gap = 0.03
    group_gap = 0.3

    group_width = n_model * bar_width + (n_model - 1) * inner_gap
    x_positions = np.arange(n_metric) * (group_width + group_gap)

    palette = sns.color_palette("Set2", n_model)

    # ===== 创建画布 =====
    plt.figure(figsize=(10, 6))

    # ===== 绘制柱状图 =====
    for i, model in enumerate(order):

        values = df[df["Model"] == model][metrics].values.flatten()

        bar_x = x_positions + i * (bar_width + inner_gap)

        plt.bar(
            bar_x,
            values,
            width=bar_width,
            label=model,
            color=palette[i],
            edgecolor="black",
            linewidth=0.6
        )

    # ===== X轴 =====
    plt.xticks(x_positions + group_width / 2 - bar_width / 2, metrics)

    plt.ylim(0.8, 0.98)
    plt.ylabel("Value")

    plt.legend(
        title="Model",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=2,
        frameon=False
    )

    plt.tight_layout()
    plt.savefig(save_path, format="svg", bbox_inches='tight')
    plt.show()


def vis_ACIM(excel_path="/home/huangdn/Causal3D-Net/src/logging_record/ACIM.xlsx",
             save_path="/home/huangdn/Causal3D-Net/src/logging_record/ACIM.svg"):

    # ===== 图形风格设置（论文推荐）=====
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 20
    plt.rcParams['axes.labelsize'] = 20
    plt.rcParams['xtick.labelsize'] = 20
    plt.rcParams['ytick.labelsize'] = 20
    plt.rcParams['legend.fontsize'] = 20
    plt.rcParams['legend.title_fontsize'] = 20
    plt.rcParams['svg.fonttype'] = 'none'
    plt.rcParams['axes.linewidth'] = 1

    # ===== 模型顺序（保持逻辑递进）=====
    order = [
        "Baseline",
        "Baseline+CCLM",
        "Baseline+ACIM",
        "Ours(CCLM+ACIM)"
    ]

    # ===== 指标 =====
    metrics = ["Acc", "AUC", "Sen", "Spe", "Prec", "F1"]

    # ===== 读取数据 =====
    df = pd.read_excel(excel_path)
    df = df[df["Model"].isin(order)]

    n_model = len(order)
    n_metric = len(metrics)

    bar_width = 0.12
    inner_gap = 0.03
    group_gap = 0.3

    group_width = n_model * bar_width + (n_model - 1) * inner_gap
    x_positions = np.arange(n_metric) * (group_width + group_gap)

    palette = sns.color_palette("Set2", n_model)

    # ===== 创建画布 =====
    plt.figure(figsize=(10, 6))

    # ===== 绘制柱状图 =====
    for i, model in enumerate(order):

        values = df[df["Model"] == model][metrics].values.flatten()

        bar_x = x_positions + i * (bar_width + inner_gap)

        plt.bar(
            bar_x,
            values,
            width=bar_width,
            label=model,
            color=palette[i],
            edgecolor="black",
            linewidth=0.6
        )

    # ===== X轴 =====
    plt.xticks(x_positions + group_width / 2 - bar_width / 2, metrics)
    plt.ylim(0.8, 0.98)
    plt.ylabel("Value")

    # ===== 图例 =====
    plt.legend(
        title="Model",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=2,
        frameon=False
    )

    plt.tight_layout()

    plt.savefig(save_path,format="svg", bbox_inches='tight')
    plt.savefig(save_path.replace(".svg", ".pdf"), bbox_inches='tight')
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
    palette[-1], palette[-2] = palette[-2], palette[-1]

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
                linewidth=0.5)

    # 设置x轴刻度在每组的中心
    plt.xticks(x_positions + group_width/2 - bar_width/2, datasets)

    # plt.ylabel("ACC")
    plt.ylim(0.65, 1)
    plt.legend(
        title="",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),  # 横向居中，纵向往下
        ncol=4,  # 图例分成多列，避免太宽
        frameon=False  # 去掉外框（可选）
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=1000, bbox_inches='tight', transparent=False)
    plt.show()


def vis_sota_model(excel_path="/home/huangdn/Causal3D-Net/src/logging_record/compare_sota.xlsx",
                   save_path="/home/huangdn/Causal3D-Net/src/logging_record/compare_sota_models.png"):
    excel = pd.read_excel(excel_path)

    metrics = ['Accuracy', 'AUC', 'Sensitivity', 'Specificity', 'Precision', 'F1']
    metric_labels = ['Acc', 'AUC', 'Sen', 'Spe', 'Prec', 'F1']  # 缩写
    datasets = ['CV', 'test I', 'test II']
    num_metrics = len(metrics)
    angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False).tolist()
    angles += angles[:1]

    ylims = {'CV': (0.1, 1), 'test I': (0.4, 1), 'test II': (0.3, 1)}

    fig, axes = plt.subplots(1, 3, figsize=(20, 6), subplot_kw=dict(polar=True))

    # 统一配色，使用 Set2，颜色数量等于方法数
    methods = excel['Methods'].unique()
    n_colors = len(methods)
    palette = sns.color_palette("Set2", n_colors)
    color_dict = dict(zip(methods, palette))

    handles, labels = [], []

    for i, dataset in enumerate(datasets):
        df_subset = excel[excel['dataset'] == dataset]
        ax = axes[i]

        for _, row in df_subset.iterrows():
            values = row[metrics].tolist()
            values += values[:1]

            method = row['Methods']
            color = color_dict[method]

            line, = ax.plot(angles, values, label=method, linewidth=2, color=color)
            ax.fill(angles, values, alpha=0.1, color=color)

            if dataset == datasets[-1]:
                handles.append(line)
                labels.append(method)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_labels)

        ymin, ymax = ylims.get(dataset, (0, 1))
        ax.set_ylim(ymin, ymax)
        ax.set_yticks(np.linspace(ymin, ymax, 6))
        ax.set_yticklabels([f"{x:.1f}" for x in np.linspace(ymin, ymax, 6)])

        ax.set_title(dataset, fontsize=14)

    # 给下方图例留空间，统一一排显示
    fig.subplots_adjust(bottom=0.05)
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0),
               ncol=len(labels), fontsize=12, frameon=False)

    plt.savefig(save_path, dpi=1000, bbox_inches='tight')  # 如需保存
    plt.show()


if __name__ == '__main__':
    # vis_10_folds_cv()
    vis_CCLM()
    vis_ACIM()
    # vis_ablation()
    # vis_sota_model()
    pass





