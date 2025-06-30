# Causal3D-Net: 使用多中心校样本对比增强CT的稳定模型用于胰腺癌诊断

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


## 数据来源

### 私有数据集

<p float="left">
  <a href="http://www.wuxihospital.com" target="_blank">
    <img src="./docs/private_1.jpeg" alt="单位1" width="100" height="100" style="border-radius:50%; object-fit: cover; margin-right:10px;" />
  </a>
  <a href="https://www.wx2h.com" target="_blank">
    <img src="./docs/private_2.jpeg" alt="单位2" width="100" height="100" style="border-radius:50%; object-fit: cover; margin-right:10px;" />
  </a>
  <a href="https://www.yzsbh.com" target="_blank">
    <img src="./docs/private_3.jpeg" alt="单位3" width="100" height="100" style="border-radius:50%; object-fit: cover; margin-right:10px;" />
  </a>
  <a href="https://www.zjhospital.net" target="_blank">
    <img src="./docs/private_4.jpeg" alt="单位4" width="100" height="100" style="border-radius:50%; object-fit: cover; margin-right:10px;" />
  </a>
  <a href="https://www.zs-hospital.sh.cn/zhuanye/" target="_blank">
    <img src="./docs/private_5.jpeg" alt="单位5" width="100" height="100" style="border-radius:50%; object-fit: cover;" />
  </a>
</p>

## 预处理

### 裁剪数据

```linux
/home/huangdn/anaconda/envs/Causal3DNet/bin/python -m src.preprocessing.ROI_data \
	--input /home/huangdn/Causal3D-Net/src/dataset/dataset.xlsx \
	--outdir /home/huangdn/Causal3D-Net/src/dataset \
	--process_num 6
```

### 提取影像组学特征

```linux
/home/huangdn/anaconda/envs/Causal3DNet/bin/python -m src.preprocessing.extract_radiomics \
	--input /home/huangdn/Causal3D-Net/src/dataset/dataset_for_radiomics_read.xlsx \
	--output /home/huangdn/Causal3D-Net/src/dataset/radiomics_features.xlsx \
	--process_num 4
```

## 训练

### 分割训练

```linux
/home/huangdn/anaconda/envs/Causal3DNet/bin/python -m src.training.train_Causal3DNet \
    --train /home/huangdn/Causal3D-Net/src/dataset/dataset_for_train.xlsx \
    --test /home/huangdn/Causal3D-Net/src/dataset/dataset_for_test.xlsx \
    --cuda 5 \
    --outdir /home/huangdn/Causal3D-Net/src/results
```

### 分类训练

```linux
/home/huangdn/anaconda/envs/Causal3DNet/bin/python -m src.training.train_Causal3DNet \
    --train /home/huangdn/Causal3D-Net/src/dataset/dataset_for_train.xlsx \
    --test /home/huangdn/Causal3D-Net/src/dataset/dataset_for_test.xlsx \
    --cuda 5 \
    --outdir /home/huangdn/Causal3D-Net/src/results \
    --weight /home/huangdn/Causal3D-Net/src/results/2025-06-21_06-49-30/last_model.pth
```

### 消融实验

#### 仅主分支

```linux
/home/huangdn/anaconda/envs/Causal3DNet/bin/python -m src.training.train_Causal3DNet \
    --train /home/huangdn/Causal3D-Net/src/dataset/dataset_for_train.xlsx \
    --test /home/huangdn/Causal3D-Net/src/dataset/dataset_for_test.xlsx \
    --indi 0 \
    --cent 0 \
    --cuda 5 \
    --outdir /home/huangdn/Causal3D-Net/src/results \
    --weight /home/huangdn/Causal3D-Net/src/results/2025-06-21_06-49-30/last_model.pth
```

#### 主分支与个人分支

```linux
/home/huangdn/anaconda/envs/Causal3DNet/bin/python -m src.training.train_Causal3DNet \
    --train /home/huangdn/Causal3D-Net/src/dataset/dataset_for_train.xlsx \
    --test /home/huangdn/Causal3D-Net/src/dataset/dataset_for_test.xlsx \
    --indi 1 \
    --cent 0 \
    --cuda 5 \
    --outdir /home/huangdn/Causal3D-Net/src/results \
    --weight /home/huangdn/Causal3D-Net/src/results/2025-06-21_06-49-30/last_model.pth
```

#### 主分支与中心分支

```linux
/home/huangdn/anaconda/envs/Causal3DNet/bin/python -m src.training.train_Causal3DNet \
    --train /home/huangdn/Causal3D-Net/src/dataset/dataset_for_train.xlsx \
    --test /home/huangdn/Causal3D-Net/src/dataset/dataset_for_test.xlsx \
    --indi 0 \
    --cent 1 \
    --cuda 5 \
    --outdir /home/huangdn/Causal3D-Net/src/results \
    --weight /home/huangdn/Causal3D-Net/src/results/2025-06-21_06-49-30/last_model.pth
```

## 对比模型

```
/home/huangdn/anaconda/envs/Causal3DNet/bin/python -m src.baselines.baseline_entry \
    --train /home/huangdn/Causal3D-Net/src/dataset/dataset_for_train.xlsx \
    --test /home/huangdn/Causal3D-Net/src/dataset/dataset_for_test.xlsx \
    --method 2.5d_vgg \
    --cuda_id 5 \
    --outdir /home/huangdn/Causal3D-Net/src/results
```

