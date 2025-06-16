# -*- coding: utf-8 -*-
# @Time    : 2025/6/9 21:28
# @Author  : D.N. Huang
# @Email   : CarlCypress@yeah.net
# @FileName: visualize.py
# @Project : Causal3D-Net
import os
import inspect
from src.augmentation.window import Windowing
from src.augmentation.brightness import MultiplicativeBrightnessTransform
from src.augmentation.contrast import ContrastTransform
from src.augmentation.gamma import GammaTransform
from src.augmentation.gaussian_blur import GaussianBlurTransform
from src.augmentation.gaussian_noise import GaussianNoiseTransform
from src.augmentation.low_resolution import SimulateLowResolutionTransform
from src.dataset.PC_dataset import PCDataset
from src.utils.visual3D import visualize_prediction, save_slices
import torchio as tio


def show_window(excel_path="", save_dir=None, idx=0):
    current_func_name = inspect.currentframe().f_code.co_name
    save_path = os.path.join(save_dir, current_func_name)
    os.makedirs(os.path.join(save_path, "original"), exist_ok=True)
    os.makedirs(os.path.join(save_path, "transformed"), exist_ok=True)

    transform = tio.Compose([
        Windowing(window_center=70, window_width=340),
    ])
    # 创建不应用变换的数据集（获取原始图像）
    orig_dataset = PCDataset(excel_path=excel_path, return_type=2, transform=None)
    trans_dataset = PCDataset(excel_path=excel_path, return_type=2, transform=transform)

    # 获取相同索引的图像
    orig_image, orig_y_sgs, orig_y_cls= orig_dataset[idx]
    trans_image, trans_y_sgs, trans_y_cls = trans_dataset[idx]

    # 转为 numpy 格式并 squeeze 掉通道维
    orig_np = orig_image.squeeze(0).numpy()  # (D, H, W)
    trans_np = trans_image.squeeze(0).numpy()

    # 保存每一层
    save_slices(orig_np, os.path.join(save_path, "original"), "original")
    save_slices(trans_np, os.path.join(save_path, "transformed"), "transformed")


def show_Resize(excel_path="", save_dir=None, idx=0):
    current_func_name = inspect.currentframe().f_code.co_name
    save_path = os.path.join(save_dir, current_func_name)
    os.makedirs(os.path.join(save_path, "original"), exist_ok=True)
    os.makedirs(os.path.join(save_path, "transformed"), exist_ok=True)

    transform = tio.Compose([
        Windowing(window_center=70, window_width=340),
        tio.RescaleIntensity(out_min_max=(0, 1)),
        tio.Resize((40, 160, 256)),
    ])
    # 创建不应用变换的数据集（获取原始图像）
    orig_dataset = PCDataset(excel_path=excel_path, return_type=2, transform=None)
    trans_dataset = PCDataset(excel_path=excel_path, return_type=2, transform=transform)

    # 获取相同索引的图像
    orig_image, orig_y_sgs, orig_y_cls= orig_dataset[idx]
    trans_image, trans_y_sgs, trans_y_cls = trans_dataset[idx]

    # 转为 numpy 格式并 squeeze 掉通道维
    orig_np = orig_image.squeeze(0).numpy()  # (D, H, W)
    trans_np = trans_image.squeeze(0).numpy()

    # 保存每一层
    save_slices(orig_np, os.path.join(save_path, "original"), "original")
    save_slices(trans_np, os.path.join(save_path, "transformed"), "transformed")


def show_GaussianNoiseTransform(excel_path="", save_dir=None, idx=0):
    current_func_name = inspect.currentframe().f_code.co_name
    save_path = os.path.join(save_dir, current_func_name)
    os.makedirs(os.path.join(save_path, "original"), exist_ok=True)
    os.makedirs(os.path.join(save_path, "transformed"), exist_ok=True)

    base_transform = tio.Compose([
        Windowing(window_center=70, window_width=340),
        tio.RescaleIntensity(out_min_max=(0, 1)),
        tio.Resize((40, 160, 256)),
    ])
    transform = tio.Compose([
        GaussianNoiseTransform(
            noise_variance=(0, 0.5),
            p_per_channel=1.0,
            synchronize_channels=True,
            p=1
        ),
    ])
    transform = tio.Compose([
        *base_transform.transforms,
        *transform.transforms,
    ])
    # 创建不应用变换的数据集（获取原始图像）
    orig_dataset = PCDataset(excel_path=excel_path, return_type=2, transform=base_transform)
    trans_dataset = PCDataset(excel_path=excel_path, return_type=2, transform=transform)

    # 获取相同索引的图像
    orig_image, orig_y_sgs, orig_y_cls= orig_dataset[idx]
    trans_image, trans_y_sgs, trans_y_cls = trans_dataset[idx]

    # 转为 numpy 格式并 squeeze 掉通道维
    orig_np = orig_image.squeeze(0).numpy()  # (D, H, W)
    trans_np = trans_image.squeeze(0).numpy()

    # 保存每一层
    save_slices(orig_np, os.path.join(save_path, "original"), "original")
    save_slices(trans_np, os.path.join(save_path, "transformed"), "transformed")


def show_GaussianBlurTransform(excel_path="", save_dir=None, idx=0):
    current_func_name = inspect.currentframe().f_code.co_name
    save_path = os.path.join(save_dir, current_func_name)
    os.makedirs(os.path.join(save_path, "original"), exist_ok=True)
    os.makedirs(os.path.join(save_path, "transformed"), exist_ok=True)

    base_transform = tio.Compose([
        Windowing(window_center=70, window_width=340),
        tio.RescaleIntensity(out_min_max=(0, 1)),
        tio.Resize((40, 160, 256)),
    ])
    transform = tio.Compose([
        GaussianBlurTransform(
            blur_sigma=(2, 5),  # 增大模糊范围
            synchronize_channels=False,
            synchronize_axes=False,
            p_per_channel=1,
            p=1
        )
    ])
    transform = tio.Compose([
        *base_transform.transforms,
        *transform.transforms,
    ])
    # 创建不应用变换的数据集（获取原始图像）
    orig_dataset = PCDataset(excel_path=excel_path, return_type=2, transform=base_transform)
    trans_dataset = PCDataset(excel_path=excel_path, return_type=2, transform=transform)

    # 获取相同索引的图像
    orig_image, orig_y_sgs, orig_y_cls= orig_dataset[idx]
    trans_image, trans_y_sgs, trans_y_cls = trans_dataset[idx]

    # 转为 numpy 格式并 squeeze 掉通道维
    orig_np = orig_image.squeeze(0).numpy()  # (D, H, W)
    trans_np = trans_image.squeeze(0).numpy()

    # 保存每一层
    save_slices(orig_np, os.path.join(save_path, "original"), "original")
    save_slices(trans_np, os.path.join(save_path, "transformed"), "transformed")


def show_MultiplicativeBrightnessTransform(excel_path="", save_dir=None, idx=0):
    current_func_name = inspect.currentframe().f_code.co_name
    save_path = os.path.join(save_dir, current_func_name)
    os.makedirs(os.path.join(save_path, "original"), exist_ok=True)
    os.makedirs(os.path.join(save_path, "transformed"), exist_ok=True)

    base_transform = tio.Compose([
        Windowing(window_center=70, window_width=340),
        tio.RescaleIntensity(out_min_max=(0, 1)),
        tio.Resize((40, 160, 256)),
    ])
    transform = tio.Compose([
        MultiplicativeBrightnessTransform(
            multiplier_range=(3, 5),
            synchronize_channels=False,
            p_per_channel=1.0,
            p=1
        ),
    ])
    transform = tio.Compose([
        *base_transform.transforms,
        *transform.transforms,
    ])
    # 创建不应用变换的数据集（获取原始图像）
    orig_dataset = PCDataset(excel_path=excel_path, return_type=2, transform=base_transform)
    trans_dataset = PCDataset(excel_path=excel_path, return_type=2, transform=transform)

    # 获取相同索引的图像
    orig_image, orig_y_sgs, orig_y_cls= orig_dataset[idx]
    trans_image, trans_y_sgs, trans_y_cls = trans_dataset[idx]

    # 转为 numpy 格式并 squeeze 掉通道维
    orig_np = orig_image.squeeze(0).numpy()  # (D, H, W)
    trans_np = trans_image.squeeze(0).numpy()

    # 保存每一层
    save_slices(orig_np, os.path.join(save_path, "original"), "original")
    save_slices(trans_np, os.path.join(save_path, "transformed"), "transformed")


def show_ContrastTransform(excel_path="", save_dir=None, idx=0):
    current_func_name = inspect.currentframe().f_code.co_name
    save_path = os.path.join(save_dir, current_func_name)
    os.makedirs(os.path.join(save_path, "original"), exist_ok=True)
    os.makedirs(os.path.join(save_path, "transformed"), exist_ok=True)

    base_transform = tio.Compose([
        Windowing(window_center=70, window_width=340),
        tio.RescaleIntensity(out_min_max=(0, 1)),
        tio.Resize((40, 160, 256)),
    ])
    transform = tio.Compose([
        ContrastTransform(
            contrast_range=(1.5, 2.5),
            preserve_range=True,
            synchronize_channels=False,
            p_per_channel=1.0,
            p=1
        ),
    ])
    transform = tio.Compose([
        *base_transform.transforms,
        *transform.transforms,
    ])
    # 创建不应用变换的数据集（获取原始图像）
    orig_dataset = PCDataset(excel_path=excel_path, return_type=2, transform=base_transform)
    trans_dataset = PCDataset(excel_path=excel_path, return_type=2, transform=transform)

    # 获取相同索引的图像
    orig_image, orig_y_sgs, orig_y_cls= orig_dataset[idx]
    trans_image, trans_y_sgs, trans_y_cls = trans_dataset[idx]

    # 转为 numpy 格式并 squeeze 掉通道维
    orig_np = orig_image.squeeze(0).numpy()  # (D, H, W)
    trans_np = trans_image.squeeze(0).numpy()

    # 保存每一层
    save_slices(orig_np, os.path.join(save_path, "original"), "original")
    save_slices(trans_np, os.path.join(save_path, "transformed"), "transformed")


def show_SimulateLowResolutionTransform(excel_path="", save_dir=None, idx=0):
    current_func_name = inspect.currentframe().f_code.co_name
    save_path = os.path.join(save_dir, current_func_name)
    os.makedirs(os.path.join(save_path, "original"), exist_ok=True)
    os.makedirs(os.path.join(save_path, "transformed"), exist_ok=True)

    base_transform = tio.Compose([
        Windowing(window_center=70, window_width=340),
        tio.RescaleIntensity(out_min_max=(0, 1)),
        tio.Resize((40, 160, 256)),
    ])
    transform = tio.Compose([
        SimulateLowResolutionTransform(
            scale=(0.2, 0.5),
            synchronize_channels=False,
            synchronize_axes=True,
            ignore_axes=(0,),
            allowed_channels=None,
            p_per_channel=1,
            p=1
        ),
    ])
    transform = tio.Compose([
        *base_transform.transforms,
        *transform.transforms,
    ])
    # 创建不应用变换的数据集（获取原始图像）
    orig_dataset = PCDataset(excel_path=excel_path, return_type=2, transform=base_transform)
    trans_dataset = PCDataset(excel_path=excel_path, return_type=2, transform=transform)

    # 获取相同索引的图像
    orig_image, orig_y_sgs, orig_y_cls= orig_dataset[idx]
    trans_image, trans_y_sgs, trans_y_cls = trans_dataset[idx]

    # 转为 numpy 格式并 squeeze 掉通道维
    orig_np = orig_image.squeeze(0).numpy()  # (D, H, W)
    trans_np = trans_image.squeeze(0).numpy()

    # 保存每一层
    save_slices(orig_np, os.path.join(save_path, "original"), "original")
    save_slices(trans_np, os.path.join(save_path, "transformed"), "transformed")


def show_GammaTransform(excel_path="", save_dir=None, idx=0):
    current_func_name = inspect.currentframe().f_code.co_name
    save_path = os.path.join(save_dir, current_func_name)
    os.makedirs(os.path.join(save_path, "original"), exist_ok=True)
    os.makedirs(os.path.join(save_path, "transformed"), exist_ok=True)

    base_transform = tio.Compose([
        Windowing(window_center=70, window_width=340),
        tio.RescaleIntensity(out_min_max=(0, 1)),
        tio.Resize((40, 160, 256)),
    ])
    transform = tio.Compose([
        GammaTransform(
            gamma=(0.7, 1.5),
            p_invert_image=1.0,
            synchronize_channels=False,
            p_per_channel=1.0,
            p_retain_stats=1.0,
            p=1
        ),
    ])
    transform = tio.Compose([
        *base_transform.transforms,
        *transform.transforms,
    ])
    # 创建不应用变换的数据集（获取原始图像）
    orig_dataset = PCDataset(excel_path=excel_path, return_type=2, transform=base_transform)
    trans_dataset = PCDataset(excel_path=excel_path, return_type=2, transform=transform)

    # 获取相同索引的图像
    orig_image, orig_y_sgs, orig_y_cls= orig_dataset[idx]
    trans_image, trans_y_sgs, trans_y_cls = trans_dataset[idx]

    # 转为 numpy 格式并 squeeze 掉通道维
    orig_np = orig_image.squeeze(0).numpy()  # (D, H, W)
    trans_np = trans_image.squeeze(0).numpy()

    # 保存每一层
    save_slices(orig_np, os.path.join(save_path, "original"), "original")
    save_slices(trans_np, os.path.join(save_path, "transformed"), "transformed")


def show_RandomAffine(excel_path="", save_dir=None, idx=0):
    current_func_name = inspect.currentframe().f_code.co_name
    save_path = os.path.join(save_dir, current_func_name)
    os.makedirs(os.path.join(save_path, "original"), exist_ok=True)
    os.makedirs(os.path.join(save_path, "transformed"), exist_ok=True)

    base_transform = tio.Compose([
        Windowing(window_center=70, window_width=340),
        tio.RescaleIntensity(out_min_max=(0, 1)),
        tio.Resize((40, 160, 256)),
    ])
    transform = tio.Compose([
        tio.RandomAffine(
            scales=(0.8, 1.2),
            degrees=10,
            isotropic=False,
            p=1),
    ])
    transform = tio.Compose([
        *base_transform.transforms,
        *transform.transforms,
    ])
    # 创建不应用变换的数据集（获取原始图像）
    orig_dataset = PCDataset(excel_path=excel_path, return_type=2, transform=base_transform)
    trans_dataset = PCDataset(excel_path=excel_path, return_type=2, transform=transform)

    # 获取相同索引的图像
    orig_image, orig_y_sgs, orig_y_cls= orig_dataset[idx]
    trans_image, trans_y_sgs, trans_y_cls = trans_dataset[idx]

    # 转为 numpy 格式并 squeeze 掉通道维
    orig_np = orig_image.squeeze(0).numpy()  # (D, H, W)
    trans_np = trans_image.squeeze(0).numpy()

    # 保存每一层
    save_slices(orig_np, os.path.join(save_path, "original"), "original")
    save_slices(trans_np, os.path.join(save_path, "transformed"), "transformed")


def show_RandomElasticDeformation(excel_path="", save_dir=None, idx=0):
    current_func_name = inspect.currentframe().f_code.co_name
    save_path = os.path.join(save_dir, current_func_name)
    os.makedirs(os.path.join(save_path, "original"), exist_ok=True)
    os.makedirs(os.path.join(save_path, "transformed"), exist_ok=True)

    base_transform = tio.Compose([
        Windowing(window_center=70, window_width=540),
        tio.RescaleIntensity(out_min_max=(0, 1)),
        tio.Resize((40, 160, 256)),
    ])
    transform = tio.Compose([
        tio.RandomElasticDeformation(
            num_control_points=9,
            max_displacement=5,
            p=1)
    ])
    transform = tio.Compose([
        *base_transform.transforms,
        *transform.transforms,
    ])
    # 创建不应用变换的数据集（获取原始图像）
    orig_dataset = PCDataset(excel_path=excel_path, return_type=2, transform=base_transform)
    trans_dataset = PCDataset(excel_path=excel_path, return_type=2, transform=transform)

    # 获取相同索引的图像
    orig_image, orig_y_sgs, orig_y_cls= orig_dataset[idx]
    trans_image, trans_y_sgs, trans_y_cls = trans_dataset[idx]
    print(orig_image.shape, trans_image.shape)

    # 转为 numpy 格式并 squeeze 掉通道维
    orig_np = orig_image.squeeze(0).numpy()  # (D, H, W)
    trans_np = trans_image.squeeze(0).numpy()

    # 保存每一层
    save_slices(orig_np, os.path.join(save_path, "original"), "original")
    save_slices(trans_np, os.path.join(save_path, "transformed"), "transformed")


def show_RandomMotion(excel_path="", save_dir=None, idx=0):
    current_func_name = inspect.currentframe().f_code.co_name
    save_path = os.path.join(save_dir, current_func_name)
    os.makedirs(os.path.join(save_path, "original"), exist_ok=True)
    os.makedirs(os.path.join(save_path, "transformed"), exist_ok=True)

    base_transform = tio.Compose([
        Windowing(window_center=70, window_width=340),
        tio.RescaleIntensity(out_min_max=(0, 1)),
        tio.Resize((40, 160, 256)),
    ])
    transform = tio.Compose([
        tio.RandomMotion(
            degrees=10,  # 最大旋转角度（单位：度）
            translation=10,  # 最大平移距离（单位：像素）
            num_transforms=2,  # 应用几个运动事件
            image_interpolation='linear',
            p=1)
    ])
    transform = tio.Compose([
        *base_transform.transforms,
        *transform.transforms,
    ])
    # 创建不应用变换的数据集（获取原始图像）
    orig_dataset = PCDataset(excel_path=excel_path, return_type=2, transform=base_transform)
    trans_dataset = PCDataset(excel_path=excel_path, return_type=2, transform=transform)

    # 获取相同索引的图像
    orig_image, orig_y_sgs, orig_y_cls= orig_dataset[idx]
    trans_image, trans_y_sgs, trans_y_cls = trans_dataset[idx]
    print(orig_image.shape, trans_image.shape)

    # 转为 numpy 格式并 squeeze 掉通道维
    orig_np = orig_image.squeeze(0).numpy()  # (D, H, W)
    trans_np = trans_image.squeeze(0).numpy()

    # 保存每一层
    save_slices(orig_np, os.path.join(save_path, "original"), "original")
    save_slices(trans_np, os.path.join(save_path, "transformed"), "transformed")


def show_RandomAnisotropy(excel_path="", save_dir=None, idx=0):
    current_func_name = inspect.currentframe().f_code.co_name
    save_path = os.path.join(save_dir, current_func_name)
    os.makedirs(os.path.join(save_path, "original"), exist_ok=True)
    os.makedirs(os.path.join(save_path, "transformed"), exist_ok=True)

    base_transform = tio.Compose([
        Windowing(window_center=70, window_width=340),
        tio.RescaleIntensity(out_min_max=(0, 1)),
        tio.Resize((40, 160, 256)),
    ])
    transform = tio.Compose([
        tio.RandomAnisotropy(
            downsampling=(1.8, 3.0),
            axes=(2,),
            p=1,
        )
    ])
    transform = tio.Compose([
        *base_transform.transforms,
        *transform.transforms,
    ])
    # 创建不应用变换的数据集（获取原始图像）
    orig_dataset = PCDataset(excel_path=excel_path, return_type=2, transform=base_transform)
    trans_dataset = PCDataset(excel_path=excel_path, return_type=2, transform=transform)

    # 获取相同索引的图像
    orig_image, orig_y_sgs, orig_y_cls= orig_dataset[idx]
    trans_image, trans_y_sgs, trans_y_cls = trans_dataset[idx]
    print(orig_image.shape, trans_image.shape)

    # 转为 numpy 格式并 squeeze 掉通道维
    orig_np = orig_image.squeeze(0).numpy()  # (D, H, W)
    trans_np = trans_image.squeeze(0).numpy()

    # 保存每一层
    save_slices(orig_np, os.path.join(save_path, "original"), "original")
    save_slices(trans_np, os.path.join(save_path, "transformed"), "transformed")


def show_RandomBiasField(excel_path="", save_dir=None, idx=0):
    current_func_name = inspect.currentframe().f_code.co_name
    save_path = os.path.join(save_dir, current_func_name)
    os.makedirs(os.path.join(save_path, "original"), exist_ok=True)
    os.makedirs(os.path.join(save_path, "transformed"), exist_ok=True)

    base_transform = tio.Compose([
        Windowing(window_center=70, window_width=340),
        tio.RescaleIntensity(out_min_max=(0, 1)),
        tio.Resize((40, 160, 256)),
    ])
    transform = tio.Compose([
        tio.RandomBiasField(
            coefficients=0.5,
            order=3,
            p=1
        )
    ])
    transform = tio.Compose([
        *base_transform.transforms,
        *transform.transforms,
    ])
    # 创建不应用变换的数据集（获取原始图像）
    orig_dataset = PCDataset(excel_path=excel_path, return_type=2, transform=base_transform)
    trans_dataset = PCDataset(excel_path=excel_path, return_type=2, transform=transform)

    # 获取相同索引的图像
    orig_image, orig_y_sgs, orig_y_cls= orig_dataset[idx]
    trans_image, trans_y_sgs, trans_y_cls = trans_dataset[idx]
    print(orig_image.shape, trans_image.shape)

    # 转为 numpy 格式并 squeeze 掉通道维
    orig_np = orig_image.squeeze(0).numpy()  # (D, H, W)
    trans_np = trans_image.squeeze(0).numpy()

    # 保存每一层
    save_slices(orig_np, os.path.join(save_path, "original"), "original")
    save_slices(trans_np, os.path.join(save_path, "transformed"), "transformed")


def show_RandomSpike(excel_path="", save_dir=None, idx=0):
    current_func_name = inspect.currentframe().f_code.co_name
    save_path = os.path.join(save_dir, current_func_name)
    os.makedirs(os.path.join(save_path, "original"), exist_ok=True)
    os.makedirs(os.path.join(save_path, "transformed"), exist_ok=True)

    base_transform = tio.Compose([
        Windowing(window_center=70, window_width=340),
        tio.RescaleIntensity(out_min_max=(0, 1)),
        tio.Resize((40, 160, 256)),
    ])
    transform = tio.Compose([
        tio.RandomSpike(
            num_spikes=3,
            intensity=(0.05, 0.15),  # 温和的伪影效果
            p=1,
        )
    ])
    transform = tio.Compose([
        *base_transform.transforms,
        *transform.transforms,
    ])
    # 创建不应用变换的数据集（获取原始图像）
    orig_dataset = PCDataset(excel_path=excel_path, return_type=2, transform=base_transform)
    trans_dataset = PCDataset(excel_path=excel_path, return_type=2, transform=transform)

    # 获取相同索引的图像
    orig_image, orig_y_sgs, orig_y_cls= orig_dataset[idx]
    trans_image, trans_y_sgs, trans_y_cls = trans_dataset[idx]
    print(orig_image.shape, trans_image.shape)

    # 转为 numpy 格式并 squeeze 掉通道维
    orig_np = orig_image.squeeze(0).numpy()  # (D, H, W)
    trans_np = trans_image.squeeze(0).numpy()

    # 保存每一层
    save_slices(orig_np, os.path.join(save_path, "original"), "original")
    save_slices(trans_np, os.path.join(save_path, "transformed"), "transformed")


def show_RandomSwap(excel_path="", save_dir=None, idx=0):
    current_func_name = inspect.currentframe().f_code.co_name
    save_path = os.path.join(save_dir, current_func_name)
    os.makedirs(os.path.join(save_path, "original"), exist_ok=True)
    os.makedirs(os.path.join(save_path, "transformed"), exist_ok=True)

    base_transform = tio.Compose([
        Windowing(window_center=70, window_width=340),
        tio.RescaleIntensity(out_min_max=(0, 1)),
        tio.Resize((40, 160, 256)),
    ])
    transform = tio.Compose([
        tio.RandomSwap(
            patch_size=15,  # 块大小（15个体素边长的立方体）
            num_iterations=100,  # 迭代次数（交换次数）
            p=1,  # 一半的概率应用
        ),
    ])
    transform = tio.Compose([
        *base_transform.transforms,
        *transform.transforms,
    ])
    # 创建不应用变换的数据集（获取原始图像）
    orig_dataset = PCDataset(excel_path=excel_path, return_type=2, transform=base_transform)
    trans_dataset = PCDataset(excel_path=excel_path, return_type=2, transform=transform)

    # 获取相同索引的图像
    orig_image, orig_y_sgs, orig_y_cls= orig_dataset[idx]
    trans_image, trans_y_sgs, trans_y_cls = trans_dataset[idx]
    print(orig_image.shape, trans_image.shape)

    # 转为 numpy 格式并 squeeze 掉通道维
    orig_np = orig_image.squeeze(0).numpy()  # (D, H, W)
    trans_np = trans_image.squeeze(0).numpy()

    # 保存每一层
    save_slices(orig_np, os.path.join(save_path, "original"), "original")
    save_slices(trans_np, os.path.join(save_path, "transformed"), "transformed")


if __name__ == '__main__':
    sdir = "/home/huangdn/Causal3D-Net/src/augmentation/visualize_results"
    epath = "/home/huangdn/Causal3D-Net/src/dataset/train_dataset.xlsx"
    show_RandomElasticDeformation(excel_path=epath, save_dir=sdir)

    pass
