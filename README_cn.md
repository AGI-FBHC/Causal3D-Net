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

模型基于因果理论学习到了专业放射科医师诊断时的主要诊断依据作为模型因果特征。

## 模型指标展示

### 与最先进的方法进行比较

#### 内部测试集

| 方法                 | Accuracy | AUC    | Sensitivity | Specificity | Precision | F1     |
| -------------------- | -------- | ------ | ----------- | ----------- | --------- | ------ |
| Radiomics            | 0.9515   | 0.9966 | 0.9247      | 0.9861      | 0.9885    | 0.9556 |
| 2.5D VGG             | 0.9152   | 0.9757 | 0.9140      | 0.9167      | 0.9341    | 0.9239 |
| ViT                  | 0.4364   | 0.5000 | 0.0000      | 1.0000      | 0.0000    | 0.0000 |
| 3D CNN               | 0.9333   | 0.9803 | 0.9570      | 0.9028      | 0.9271    | 0.9418 |
| Hybrid  Transformer  | 0.6727   | 0.7446 | 0.5161      | 0.8750      | 0.8421    | 0.6400 |
| C Net                | 0.4364   | 0.5000 | 0.0000      | 1.0000      | 0.0000    | 0.0000 |
| Neural  Transformers |          |        |             |             |           |        |
| Causal3DNet(Ours)    | 0.9394   | 0.9714 | 0.9355      | 0.9444      | 0.9560    | 0.9457 |

#### 外部测试集1

| 方法                 | Accuracy | AUC    | Sensitivity | Specificity | Precision | F1     |
| -------------------- | -------- | ------ | ----------- | ----------- | --------- | ------ |
| Radiomics            | 0.7875   | 0.8701 | 0.8625      | 0.7125      | 0.7500    | 0.8023 |
| 2.5D VGG             | 0.8750   | 0.9440 | 0.8750      | 0.8750      | 0.8750    | 0.8750 |
| ViT                  | 0.5000   | 0.5000 | 0.0000      | 1.0000      | 0.0000    | 0.0000 |
| 3D CNN               | 0.7000   | 0.7778 | 0.6750      | 0.7250      | 0.7105    | 0.6923 |
| Hybrid  Transformer  | 0.4688   | 0.4292 | 0.6000      | 0.3375      | 0.4752    | 0.5304 |
| C Net                | 0.5000   | 0.5000 | 0.0000      | 1.0000      | 0.0000    | 0.0000 |
| Neural  Transformers |          |        |             |             |           |        |
| Causal3DNet(Ours)    | 0.9062   | 0.9605 | 0.8375      | 0.9750      | 0.9710    | 0.8993 |

#### 外部测试集2

| 方法                 | Accuracy | AUC  | Sensitivity | Specificity | Precision | F1   |
| -------------------- | -------- | ---- | ----------- | ----------- | --------- | ---- |
| Radiomics            | 0.8033   |      |             | 0.8033      |           |      |
| 2.5D VGG             | 0.6557   |      |             | 0.6557      |           |      |
| ViT                  | 1.0000   |      |             | 1.0000      |           |      |
| 3D CNN               | 0.7541   |      |             | 0.7541      |           |      |
| Hybrid  Transformer  | 0.7869   |      |             | 0.7869      |           |      |
| C Net                | 1.0000   |      |             | 1.0000      |           |      |
| Neural  Transformers |          |      |             |             |           |      |
| Causal3DNet(Ours)    | 0.9344   |      |             | 0.9344      |           |      |

#### 外部测试集3

| 方法                 | Accuracy | AUC  | Sensitivity | Specificity | Precision | F1   |
| -------------------- | -------- | ---- | ----------- | ----------- | --------- | ---- |
| Radiomics            | 0.7257   |      |             | 0.7257      |           |      |
| 2.5D VGG             | 0.5212   |      |             | 0.5212      |           |      |
| ViT                  | 1.0000   |      |             | 1.0000      |           |      |
| 3D CNN               | 0.6646   |      |             | 0.6646      |           |      |
| Hybrid  Transformer  | 0.5075   |      |             | 0.5075      |           |      |
| C Net                | 1.0000   |      |             | 1.0000      |           |      |
| Neural  Transformers |          |      |             |             |           |      |
| Causal3DNet(Ours)    | 0.8466   |      |             | 0.8466      |           |      |

## 安装

```linux
conda env create -f environment.yaml
```

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
