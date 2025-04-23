# Causal3D-Net: A Stable Model for Pancreatic Cancer Diagnosis via Multi-Center Small-Sample Contrast-Enhanced CT

## Preprocessing

### resample data

```linux
python ./src/preprocessing/resample_data.py --input /home/huangdn/Causal3D-Net/src/dataset/dataset.xlsx --outdir /home/huangdn/Causal3D-Net/src/data
```

### extract radiomics features

```linux
python ./src/preprocessing/individual_confounders.py --input /home/huangdn/Causal3D-Net/src/dataset/radiomics_read.xlsx --output /home/huangdn/Causal3D-Net/src/data/radiomics_features.xlsx
```





