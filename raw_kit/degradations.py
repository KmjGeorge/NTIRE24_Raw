import rawpy
import numpy as np
import glob, os
import imageio
import argparse
from PIL import Image as PILImage
import scipy.io as scio
from glob import glob
from tqdm import tqdm
from matplotlib import pyplot as plt
import cv2

from scipy.io import loadmat
from scipy import ndimage
from scipy.signal import convolve2d
import hdf5storage

import torch
import torch.nn as nn
import torch.nn.functional as F

from raw_kit.blur import apply_psf, add_blur
from raw_kit.noise import add_natural_noise, add_gnoise, add_heteroscedastic_gnoise
from raw_kit.imutils import downsample_raw, convert_to_tensor
from raw_kit.load_data import load_data


def simple_deg_simulation(img, kernels):
    """
    Pipeline to add synthetic degradations to a (RAW/RGB) image.
    y = down(x * k) + n
    """

    img = convert_to_tensor(img)
    img = img.unsqueeze(0)  # (1, 4, 512, 512)

    # Apply psf blur: x * k
    img = add_blur(img, kernels)

    # Apply downsampling down(x*k)
    img = downsample_raw(img)
    # Add noise down(x*k) + n
    p_noise = np.random.rand()
    if p_noise > 0.3:
        img = add_natural_noise(img, False)
    else:
        img = add_heteroscedastic_gnoise(img, 'cpu')

    return img


def linear_exposure_compensation(image_array, maxval, compensation_value):
    # 确保补偿值在合理的范围内
    compensation_value = max(-1, min(1, compensation_value))

    # 应用曝光补偿
    # 这里使用线性变换来调整曝光
    # 公式为: new_value = scale * (original_value - mid_value) + mid_value
    # 其中 scale 是补偿因子，mid_value 是像素值范围的中点（例如，对于 uint8 范围是 128）
    mid_value = maxval / 2  # 对于 uint8 类型的图像，中点是 128
    scale = 1 + compensation_value  # 计算缩放因子

    # 对每个颜色通道进行曝光补偿
    adjusted_image = scale * (image_array - mid_value) + mid_value

    # 确保调整后的像素值在范围内
    adjusted_image = torch.clip(adjusted_image, 0, maxval)

    return adjusted_image


if __name__ == '__main__':

    raw, maxval = load_data('D:\Datasets\DSLR\\val_dev\\val_in\\1.npz')
    cv2.imshow('raw', raw)
    raw = torch.from_numpy(raw).permute(2, 0, 1).unsqueeze(0)
    '''
    kernels = np.load('kernels.npy', allow_pickle=True)
    deg_raw = simple_deg_simulation(raw, kernels).detach().numpy()
    print(raw.shape, deg_raw.shape)
    print(raw.max(), raw.min())
    print(deg_raw.max(), deg_raw.min())
    '''
    raw_exp = linear_exposure_compensation(raw, 1, compensation_value=0.3)
    raw_exp_img = raw_exp.detach().squeeze(0).permute(1, 2, 0).numpy()
    cv2.imshow('after', raw_exp_img)
    cv2.waitKey(0)
