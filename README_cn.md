# Causal3D-Net: 使用多中心小样本对比增强CT的稳定模型用于胰腺癌诊断

<p align="center">
  <a href="README.md">English</a> |
  <a href="README_cn.md">中文</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/OS-Ubuntu%2022.04-blue" />
  <img src="https://img.shields.io/badge/Python-3.10+-red" />
  <img src="https://img.shields.io/badge/Framework-PyTorch-orange" />
  <img src="https://img.shields.io/badge/Build-Success-brightgreen" />
  <img src="https://img.shields.io/badge/License-MIT-blue" />
  <img src="https://img.shields.io/badge/Release-0.1-blue" />
</p>
这是Causal3DNet的官方仓库——针对胰腺癌诊断的专用模型，

模型基于因果学习理论，让模型能够辨别因果因素和混杂因子，最后使用可靠因果特征作为诊断依据，实现稳定学习诊断，

下图展示了论文提出的模型架构图：

![model_structure](./docs/model_structure.svg)

## ⚙️ 安装

```linux
conda env create -f environment.yaml
```

## 🚀 使用

所有实验都可以使用以下方面启动：

```linux
python -m src.training.main <mode> [options]
```

### 分割训练

```linux
python -m src.training.main seg \
    --train /path/to/dataset_for_train.xlsx \
    --test /path/to/dataset_for_test.xlsx \
    --cuda 0 \
    --outdir ./results/segmentation
```

**参数：**

- --train : 训练数据集Excel文件的路径。
- --test : 测试数据集Excel文件的路径。
- --cuda : 需要使用的GPU索引（默认为0）。
- --outdir: 生成的结果保存路径（默认为./results）。

### Causal3DNet训练

```linux
python -m src.training.main causal \
    --train /path/to/dataset_for_train.xlsx \
    --test /path/to/dataset_for_test.xlsx \
    --indi 1 \
    --cent 1 \
    --orth 1 \
    --cuda 0 \
    --outdir ./results/causal3dnet \
    --weight ./pretrained/best_model.pth
```

**参数：**

- --train : 训练数据集Excel文件的路径。
- --test : 测试数据集Excel文件的路径。
- --indi : 是否使用个体分支（默认使用）。
- --cent : 是否使用中心分支（默认使用）。
- --orth : 使用使用正交约束（默认使用）。
- --cuda : 需要使用的GPU索引（默认为0）。
- --outdir: 生成的结果保存路径（默认为./results）。
- --weight: 分割预训练的权重文件路径（默认为./best_model.pth）。

## 📝 注意事项

- 使用 **-m** 来运行模块，以确保 Python 正确解析包的导入。
- 确保您的数据集已准备为 **Excel 格式**（.xlsx），并包含合适的训练集和测试集划分。
- GPU 索引（–cuda）应根据您的硬件情况进行设置。

### 数据集Excel文件样例

| image_path                   | mask_path                     | cancer | center | cluster |
| ---------------------------- | ----------------------------- | ------ | ------ | ------- |
| Center01Img00002_private.npy | Center01Mask00002_private.npy | 1      | 0      | 0       |
| Center01Img00003_private.npy | Center01Mask00003_private.npy | 1      | 0      | 0       |
| …                            | …                             | …      | …      | …       |

## **📖 引用**

如果您在研究中觉得本仓库有帮助，请考虑引用我们的工作：

```
@article{huang2025causal3dnet,
  title   = {Causal3D-Net: A Causal Model for Multi-center Small-sample Pancreatic Cancer Diagnosis Using Contrast-enhanced CT},
  author  = {Denan, Huang and Others},
  journal = {mabye TIP, TMI or MIA},
  year    = {2025 or 2026},
  volume  = {XX},
  pages   = {XX--XX},
  doi     = {10.XXXX/j.media.2025.XXXXX}
}
```

## 🤝合作单位

<p align="center">
  <a href="http://www.wuxihospital.com">
    <img src="./docs/private_1.jpeg" width="80" style="border-radius: 50%" alt="private_1"/>
  </a>
  &nbsp;&nbsp;&nbsp;
  <a href="https://www.wx2h.com">
    <img src="./docs/private_2.jpeg" width="80" style="border-radius: 50%" alt="private_2"/>
  </a>
  &nbsp;&nbsp;&nbsp;
  <a href="https://www.yzsbh.com">
    <img src="./docs/private_3.jpeg" width="80" style="border-radius: 50%" alt="private_3"/>
  </a>
  &nbsp;&nbsp;&nbsp;
  <a href="https://www.zjhospital.net">
    <img src="./docs/private_4.jpeg" width="80" style="border-radius: 50%" alt="private_4"/>
  </a>
  &nbsp;&nbsp;&nbsp;
  <a href="https://www.zs-hospital.sh.cn/zhuanye/">
    <img src="./docs/private_5.jpeg" width="80" style="border-radius: 50%" alt="private_5"/>
  </a>
</p>
