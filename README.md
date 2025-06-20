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

### stage 1 training

```linux
/home/huangdn/anaconda/envs/Causal3DNet/bin/python -m src.training.train_PANDA \
    --train /home/huangdn/Causal3D-Net/src/dataset/train_dataset.xlsx \
    --cuda 5 \
    --test /home/huangdn/Causal3D-Net/src/dataset/test_dataset.xlsx \
    --outdir /home/huangdn/Causal3D-Net/src/results
```

### stage 2 training

```
/home/huangdn/anaconda/envs/Causal3DNet/bin/python -m src.training.train_PANDA \
    --train /home/huangdn/Causal3D-Net/src/dataset/train_dataset.xlsx \
    --test /home/huangdn/Causal3D-Net/src/dataset/test_dataset.xlsx \
    --cuda 5 \
    --outdir /home/huangdn/Causal3D-Net/src/results \
    --weight /home/huangdn/Causal3D-Net/src/results/2025-06-01_05-55-00/best_model.pth
```

