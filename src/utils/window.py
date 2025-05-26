# -*- coding: utf-8 -*-
# @Time    : 2025/4/29 17:35
# @Author  : D.N. Huang
# @Email   : CarlCypress@yeah.net
# @FileName: window.py
# @Project : Causal3D-Net
import torchio as tio


# class Windowing:
#     def __init__(self, window_center=70, window_width=340):
#         self.lower = window_center - window_width // 2
#         self.upper = window_center + window_width // 2
#
#     def __call__(self, subject: tio.Subject):
#         image = subject['image']
#         image_tensor = image.data  # shape: (1, D, H, W)
#         image_tensor = image_tensor.clamp(min=self.lower, max=self.upper)
#         subject['image'].set_data(image_tensor)
#         return subject


class Windowing(tio.Transform):
    def __init__(self, window_center=70, window_width=340):
        super().__init__()
        self.lower = window_center - window_width // 2
        self.upper = window_center + window_width // 2

    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        image = subject['image']
        if isinstance(image, tio.ScalarImage):
            image_tensor = image.data
            image_tensor = image_tensor.clamp(min=self.lower, max=self.upper)
            subject['image'].set_data(image_tensor)
        return subject



