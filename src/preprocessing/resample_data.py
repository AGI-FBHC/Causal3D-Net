# -*- coding: utf-8 -*-
# @Time    : 2025/3/29 10:08
# @Author  : D.N. Huang
# @Email   : CarlCypress@yeah.net
# @FileName: resample_data.py
# @Project : Causal3D-Net
import os
import shutil
import logging
import logging.handlers
import numpy as np
import pandas as pd
import os, argparse
import nibabel as nib
from scipy.ndimage import zoom
from concurrent.futures import ProcessPoolExecutor, as_completed
from queue import Queue


def resample_z_direction(
        nii_path: str,
        is_mask: bool = False,
        output_path: str = 'output.nii.gz',
        spacing: float = 1.0,
        target_depth: int = None,
        is_print: bool = False
        ) -> int:
    """对 NIfTI 文件的 z 方向重采样，若 target_depth 不为 None，
    则根据 target_depth 变换厚度，否则根据物理距离 spacing.
    :param nii_path: 输入的 nifti 文件路径。
    :param is_mask: 是否是掩码文件?
    :param output_path: 重采样后保存的文件路径。
    :param spacing: z 轴的新体素间距(单位：mm)。
    :param target_depth: z轴采样目标厚度(单位：pixel)。
    :param is_print: 是否打印文件处理状态?
    :return: 原始影像z轴的厚度(方便逆操作)。
    """
    nii = nib.load(nii_path)
    data = nii.get_fdata()
    affine = nii.affine
    z_spacing = np.abs(affine[2, 2])
    original_depth = data.shape[2]

    if target_depth is not None:
        # Calculate the zoom factor to match the target depth
        zoom_factor = target_depth / original_depth
        spacing = z_spacing / zoom_factor  # Update spacing based on the target depth
    else:
        # Use the provided physical spacing
        zoom_factor = z_spacing / spacing

    scale_factors = [1, 1, zoom_factor]
    new_data = zoom(data, scale_factors, order=0 if is_mask else 3)  # 对image数据采用3次插值，mask数据采用最近邻插值
    new_data = np.rint(new_data).astype(np.uint8) if is_mask else new_data  # mask 可能存在差值后的精度问题，需要舍入

    new_affine = affine.copy()
    new_affine[2, 2] = np.sign(affine[2, 2]) * spacing  # Update the z-spacing in the affine matrix

    new_img = nib.Nifti1Image(new_data, affine=new_affine, header=nii.header)

    # 更新 z 轴的相关字段
    new_img.header['dim'][3] = new_data.shape[2]  # 更新 z 方向的维度
    new_img.header['pixdim'][3] = spacing  # 更新 z 轴体素间距
    new_img.header['srow_z'] = new_affine[2, :4]  # 更新 srow_z（仿射矩阵的第 3 行）

    nib.save(new_img, output_path)

    print(f'resampling {nii_path} completed.') if is_print else None
    return original_depth


def get_z_spacing_list(resample_num):
    if resample_num == 1:
        return [1]
    elif resample_num == 3:
        return [1, 3, 5]
    elif resample_num == 5:
        return [1, 2, 3, 4, 5]
    pass


def find_closest_number(lst, num):
    return min(lst, key=lambda x: (abs(x - num), x))


def setup_logger(queue, log_path):
    # 设置主进程的日志记录
    handler = logging.handlers.QueueHandler(queue)  # 处理日志队列
    root = logging.getLogger()
    root.addHandler(handler)
    root.setLevel(logging.INFO)

    # 设置日志文件输出
    file_handler = logging.FileHandler(os.path.join(log_path, 'resample_logging.log'), mode='w')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)


# def process_row(index, row, z_spacing_list, images_save_dir, masks_save_dir, data_finger, is_overwrite):
#     image_path = row['image_path']
#     mask_path = row['mask_path']
#     name_end = "public" if "public" in image_path else "private"
#     image_name = os.path.basename(image_path)
#     mask_name = os.path.basename(mask_path)
#     cancer = row['cancer']
#     image_nii = nib.load(image_path)
#     origin_spacing = abs(image_nii.affine[2, 2])
#     aligned_spacing = find_closest_number(z_spacing_list, origin_spacing)
#
#     for z_spacing in z_spacing_list:
#         new_image_path = os.path.join(images_save_dir, image_name.replace(".nii.gz", f"_{z_spacing:05d}_{name_end}.nii.gz"))
#         new_mask_path = os.path.join(masks_save_dir, mask_name.replace(".nii.gz", f"_{z_spacing:05d}_{name_end}.nii.gz"))
#         data_finger.append([new_image_path, new_mask_path, cancer, z_spacing == aligned_spacing])
#         image_continue = True if os.path.isfile(new_image_path) and not is_overwrite else False
#         mask_continue = True if os.path.isfile(new_mask_path) and not is_overwrite else False
#
#         if z_spacing == aligned_spacing:
#             shutil.copy(image_path, new_image_path) if not image_continue else None
#             logging.info(f"{new_image_path} completed.")
#             shutil.copy(mask_path, new_mask_path) if not mask_continue else None
#             logging.info(f"{new_mask_path} completed.")
#             continue
#
#         resample_z_direction(nii_path=image_path, is_mask=False, output_path=new_image_path, spacing=z_spacing) if not image_continue else None
#         logging.info(f"{new_image_path} completed.")
#         resample_z_direction(nii_path=mask_path, is_mask=True, output_path=new_mask_path, spacing=z_spacing) if not mask_continue else None
#         logging.info(f"{new_mask_path} completed.")
#
#     return data_finger  # 返回修改后的data_finger



def resample_data():
    parser = argparse.ArgumentParser(description="Resample images and masks")
    # parser.add_argument("--excel_path", type=str, default="/home/huangdn/Causal3D-Net/src/dataset/dataset.xlsx", help="Origin sorted images and masks Excel file path.")
    # parser.add_argument("--out_path", type=str, default="/home/huangdn/Causal3D-Net/src/data", help="Output resampled images and masks dir path.")
    parser.add_argument("--excel_path", type=str, required=True, help="Origin sorted images and masks Excel file path.")
    parser.add_argument("--out_path", type=str, required=True, help="Output resampled images and masks dir path.")
    # parser.add_argument("--process_num", type=int, default=2, help="Number of concurrent processes to run, be careful not to exceed the number of CPU cores.")
    parser.add_argument("--overwrite", type=bool, default=False, help="Overwrite existing resampled images and masks.")
    parser.add_argument("--resample_num", type=int, choices=[1, 3, 5], default=5, help="Total after resampling. Choose from 1, 3, or 5.")
    parser.add_argument("--log_path", type=str, default="/home/huangdn/Causal3D-Net/src/logging_record", help="Logging record path.")
    args = parser.parse_args()

    logging.basicConfig(
        filename=os.path.join(args.log_path, 'resample_logging.log'),  # 设置日志文件名
        level=logging.INFO,  # 设置日志级别为 INFO（会记录 INFO 及更高级别的日志）
        format='%(asctime)s - %(levelname)s - %(message)s',  # 日志格式
        filemode='w'
    )
    # queue = Queue()
    # setup_logger(queue, args.log_path)

    dataset_excel = pd.read_excel(args.excel_path)
    images_save_dir = os.path.join(args.out_path, "images")
    masks_save_dir = os.path.join(args.out_path, "masks")
    data_finger_save_path = os.path.join(args.out_path, "data_finger.xlsx")
    is_overwrite = args.overwrite
    os.makedirs(images_save_dir, exist_ok=True)
    os.makedirs(masks_save_dir, exist_ok=True)
    z_spacing_list = get_z_spacing_list(args.resample_num)
    data_finger = list()

    # # 使用ProcessPoolExecutor来并行处理数据
    # with ProcessPoolExecutor(max_workers=args.process_num) as executor:
    #     futures = []
    #     for index, row in dataset_excel.iterrows():
    #         futures.append(executor.submit(process_row, index, row, z_spacing_list, images_save_dir, masks_save_dir, data_finger, is_overwrite))
    #
    #     # 等待所有任务完成
    #     for future in as_completed(futures):
    #         data_finger.extend(future.result())

    for index, row in dataset_excel.iterrows():
        image_path = row['image_path']
        mask_path = row['mask_path']
        name_end = "public" if "public" in image_path else "private"
        image_name = os.path.basename(image_path)
        mask_name = os.path.basename(mask_path)
        cancer = row['cancer']
        image_nii = nib.load(image_path)
        origin_spacing = abs(image_nii.affine[2, 2])
        aligned_spacing = find_closest_number(z_spacing_list, origin_spacing)
        for z_spacing in z_spacing_list:
            new_image_path = os.path.join(images_save_dir, image_name.replace(".nii.gz", f"_{z_spacing:05d}_{name_end}.nii.gz"))
            new_mask_path = os.path.join(masks_save_dir, mask_name.replace(".nii.gz", f"_{z_spacing:05d}_{name_end}.nii.gz"))
            data_finger.append([new_image_path, new_mask_path, cancer, z_spacing == aligned_spacing])
            image_continue = True if os.path.isfile(new_image_path) and not is_overwrite else False
            mask_continue = True if os.path.isfile(new_mask_path) and not is_overwrite else False
            if z_spacing == aligned_spacing:
                shutil.copy(image_path, new_image_path) if not image_continue else None
                logging.info(f"{new_image_path} completed.")
                shutil.copy(mask_path, new_mask_path) if not mask_continue else None
                logging.info(f"{new_mask_path} completed.")
                continue
            resample_z_direction(nii_path=image_path, is_mask=False, output_path=new_image_path, spacing=z_spacing) if not image_continue else None
            logging.info(f"{new_image_path} completed.")
            resample_z_direction(nii_path=mask_path, is_mask=True, output_path=new_mask_path, spacing=z_spacing) if not mask_continue else None
            logging.info(f"{new_mask_path} completed.")
            pass
    #     if index == 1:
    #         break
    finger_df = pd.DataFrame(data_finger, columns=["image_path", "mask_path", "cancer", "raw_data"])
    finger_df.to_excel(data_finger_save_path, index=False)

    # listener = logging.handlers.QueueListener(queue, logging.getLogger())
    # listener.start()
    # listener.stop()  # 结束后停止日志监听器
    pass


if __name__ == "__main__":
    resample_data()

