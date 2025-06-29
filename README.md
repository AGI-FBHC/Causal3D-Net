# Causal3D-Net: A Stable Model for Pancreatic Cancer Diagnosis via Multi-Center Small-Sample Contrast-Enhanced CT

## Preprocessing

### crop data

```linux
/home/huangdn/anaconda/envs/Causal3DNet/bin/python -m src.preprocessing.ROI_data \
	--input /home/huangdn/Causal3D-Net/src/dataset/dataset.xlsx \
	--outdir /home/huangdn/Causal3D-Net/src/dataset \
	--process_num 6
```

### extract radiomics features

```linux
/home/huangdn/anaconda/envs/Causal3DNet/bin/python -m src.preprocessing.extract_radiomics \
	--input /home/huangdn/Causal3D-Net/src/dataset/dataset_for_radiomics_read.xlsx \
	--output /home/huangdn/Causal3D-Net/src/dataset/radiomics_features.xlsx \
	--process_num 4
```

## Training

### segmentation training

```linux
/home/huangdn/anaconda/envs/Causal3DNet/bin/python -m src.training.train_Causal3DNet \
    --train /home/huangdn/Causal3D-Net/src/dataset/dataset_for_train.xlsx \
    --test /home/huangdn/Causal3D-Net/src/dataset/dataset_for_test.xlsx \
    --cuda 5 \
    --outdir /home/huangdn/Causal3D-Net/src/results
```

### classification training

```linux
/home/huangdn/anaconda/envs/Causal3DNet/bin/python -m src.training.train_Causal3DNet \
    --train /home/huangdn/Causal3D-Net/src/dataset/dataset_for_train.xlsx \
    --test /home/huangdn/Causal3D-Net/src/dataset/dataset_for_test.xlsx \
    --cuda 5 \
    --outdir /home/huangdn/Causal3D-Net/src/results \
    --weight /home/huangdn/Causal3D-Net/src/results/2025-06-21_06-49-30/last_model.pth
```

### Ablation

#### only individual branch

```
/home/huangdn/anaconda/envs/Causal3DNet/bin/python -m src.training.train_Causal3DNet \
    --train /home/huangdn/Causal3D-Net/src/dataset/dataset_for_train.xlsx \
    --test /home/huangdn/Causal3D-Net/src/dataset/dataset_for_test.xlsx \
    --indi 1 \
    --cent 0 \
    --cuda 5 \
    --outdir /home/huangdn/Causal3D-Net/src/results \
    --weight /home/huangdn/Causal3D-Net/src/results/2025-06-21_06-49-30/last_model.pth
```

#### only center branch

```
/home/huangdn/anaconda/envs/Causal3DNet/bin/python -m src.training.train_Causal3DNet \
    --train /home/huangdn/Causal3D-Net/src/dataset/dataset_for_train.xlsx \
    --test /home/huangdn/Causal3D-Net/src/dataset/dataset_for_test.xlsx \
    --indi 0 \
    --cent 1 \
    --cuda 5 \
    --outdir /home/huangdn/Causal3D-Net/src/results \
    --weight /home/huangdn/Causal3D-Net/src/results/2025-06-21_06-49-30/last_model.pth
```

## Baseline

```
/home/huangdn/anaconda/envs/Causal3DNet/bin/python -m src.baselines.baseline_entry \
    --train /home/huangdn/Causal3D-Net/src/dataset/dataset_for_train.xlsx \
    --test /home/huangdn/Causal3D-Net/src/dataset/dataset_for_test.xlsx \
    --method 2.5d_vgg \
    --cuda_id 5 \
    --outdir /home/huangdn/Causal3D-Net/src/results
```

