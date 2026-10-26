#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 18:30:22 2026

@author: zahid
"""

import torch
import torch.nn.functional as F
import random
import numpy as np
from torchvision.transforms import functional as Ft
import math
from torchvision import datasets, transforms

# ------------------------
# Define the basic augmentations
# ------------------------

class FlipHorizontal_sync:
    def __init__(self, p=1.0):
        self.p = p

    def __call__(self, img):
        if torch.rand(1).item() < self.p:
            img = torch.flip(img, dims=[-1])  # flip width
        return img

class FlipVertical_sync:
    def __init__(self, p=1.0):
        self.p = p

    def __call__(self, img):
        if torch.rand(1).item() < self.p:
            img = torch.flip(img, dims=[-2])  # flip height
        return img
    
    
class Rotate:
    def __init__(self, angle):
        self.angle = random.randint(-angle, angle)

    def __call__(self, img):
        return Ft.rotate(img, angle=self.angle)
    
class Rotate_sync:
    def __init__(self, max_angle=30.0):
        self.max_angle = max_angle  
    
    def __call__(self, img):

         B, C, H, W = img.shape
         theta_list = []
    
         for _ in range(B):
             angle = random.uniform(-self.max_angle, self.max_angle)
             angle_rad = math.radians(angle)
             cos_a = math.cos(angle_rad)
             sin_a = math.sin(angle_rad)
    
             theta = torch.tensor([
                 [cos_a, -sin_a, 0],
                 [sin_a,  cos_a, 0]
             ], dtype=torch.float32)
             theta_list.append(theta)
    
         theta_batch = torch.stack(theta_list).to(img.device)  # [B, 2, 3]
    
         grid = F.affine_grid(theta_batch, size=img.size(), align_corners=False)
         img_rot = F.grid_sample(img.float(), grid, mode='nearest', padding_mode='zeros', align_corners=False)
    
         return img_rot
     
class GaussianNoise_torch_gpu:
    def __init__(self, mean=0, std_range=(5, 20)):
        self.mean = mean
        self.std_range = std_range

    def __call__(self, img_tensor):
        std = random.uniform(*self.std_range)
        noise = torch.randn_like(img_tensor) * std + self.mean
        img_tensor = torch.clamp(img_tensor + noise, 0, 255)
        return img_tensor
    
    
class SaltAndPepperNoise_torch_gpu:
    def __init__(self, prob_range=(0.01, 0.03)):
        self.prob_range = prob_range

    def __call__(self, img_tensor):
        B, C, H, W = img_tensor.shape
        salt_prob = random.uniform(*self.prob_range)
        pepper_prob = random.uniform(*self.prob_range)

        salt_mask = torch.rand(B, 1, H, W, device=img_tensor.device) < salt_prob
        pepper_mask = torch.rand(B, 1, H, W, device=img_tensor.device) < pepper_prob

        salt_mask = salt_mask.expand(-1, C, -1, -1)
        pepper_mask = pepper_mask.expand(-1, C, -1, -1)

        img_tensor = img_tensor.clone()
        img_tensor[salt_mask] = 255
        img_tensor[pepper_mask] = 0
        return img_tensor
    
class MotionBlur_torch_gpu:
    def __init__(self, kernel_size_range=(3, 15)):
        self.kernel_size = random.choice(range(*kernel_size_range))

    def __call__(self, img_tensor):
        B, C, H, W = img_tensor.shape
        k = self.kernel_size
        padding = (k // 2, 0)

        kernel = torch.ones(1, 1, 1, k, device=img_tensor.device) / k
        kernel = kernel.expand(C, 1, 1, k)

        blurred = F.conv2d(img_tensor.float(), kernel, padding=padding, groups=C)
        return blurred.clamp(0, 255)
    
class HideAndSeekNoise_torch_gpu:
    def __init__(self, patch_size_range=(90, 150)):
        self.patch_size = random.randint(*patch_size_range)

    def __call__(self, img_tensor):
        B, C, H, W = img_tensor.shape
        patch_size = min(self.patch_size, H - 20, W - 20)

        for i in range(B):
            top = random.randint(0, H - patch_size)
            left = random.randint(0, W - patch_size)
            img_tensor[i, :, top:top + patch_size, left:left + patch_size] = 0

        return img_tensor
    
    
class ImageOnlyTransform:
    """Wrapper to apply a function only on the image"""
    def __init__(self, transform):
        self.transform = transform
    def __call__(self, x):
        return self.transform(x)


# ------------------------
# Full Medical Batch Augmentation
# ------------------------


def _augmentation_gpu(batch_tensor, augmentation_num = 1, im_size = 512):  
    """
    Applies a sequence of randomized augmentations to a batch of images and their corresponding semantic segmentation masks.

    Args:
        batch_tensor (Tensor): A tensor of shape [B, C, H, W] representing a batch of input images.
        batch_target (Tensor): A tensor of shape [B, H, W] representing the corresponding target labels (segmentation masks).
        augmentation_num (int): The number of augmentations to randomly select and apply from the predefined set.
        im_size (int): The target size to which the augmented images and masks will be resized.

    Returns:
        Tuple[Tensor, Tensor]: A tuple containing:
            - Augmented images of shape [B, C, im_size, im_size], dtype uint8.
            - Augmented targets of shape [B, im_size, im_size], dtype long.
    """
    device = batch_tensor.device 
    
    augmentation_set = [
                FlipHorizontal_sync(p=1.0),
                FlipVertical_sync(p=1.0),
                # Crop_sync(min_crop=0.6, max_crop=0.9),   # Random crop
                Rotate_sync(max_angle= 15),  # Rotate
                # Shear_sync(shear=10, scale=(0.8, 1.2)),  # Shear
                # Skew_sync(magnitude=0.2),  # Skew
                HideAndSeekNoise_torch_gpu(patch_size_range=(20, 30)), #Hide and seek noise
        
                GaussianNoise_torch_gpu(mean=0, std_range=(5,20)), # Gaussian noise (only for image) / std 10-20
                SaltAndPepperNoise_torch_gpu(prob_range=(0.01, 0.03)), #Salt and pepper noise (only for image) 
                transforms.GaussianBlur(kernel_size=3, sigma=(0.2, 1)),  # Gaussian blur (only for image)
                MotionBlur_torch_gpu(kernel_size_range=(3, 15)), # Motion blur (only for image)
                # ImageOnlyTransform(transforms.ColorJitter(brightness=0.2, hue = 0.2)),
            ]
    
    
    
    random.shuffle(augmentation_set)
    selected_augs = augmentation_set[:augmentation_num]
    
    # print(selected_augs)
    
    
    if len(selected_augs)>0:
        for aug in selected_augs:
            batch_tensor = aug(batch_tensor)
        
    
        
    if batch_tensor.shape[-2:] != (im_size, im_size):
        batch_tensor = F.interpolate(batch_tensor, size=(im_size, im_size), mode='bilinear', align_corners=False)
        
    return batch_tensor.float().to(device)
