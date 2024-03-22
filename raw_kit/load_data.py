import os
from glob import glob
import numpy as np
import torch
import torch.nn.functional as F
import cv2

def downsample_raw(raw):
    """
    Downsamples a 4-channel packed RAW image by a factor of 2.
    The input raw should be a [H/2, W/2, 4] tensor -- with respect to its mosaiced version [H,w]
    Output is a [H/4, W/4, 4] tensor, preserving the RGGB pattern.
    """
    # Ensure the image is in [B, C, H, W] format for PyTorch operations
    raw_image_4channel = raw.permute(2, 0, 1).unsqueeze(0)
    raw = raw_image_4channel
    # Apply average pooling over a 2x2 window for each channel
    downsampled_image = F.avg_pool2d(raw, kernel_size=2, stride=2, padding=0)

    # Rearrange back to [H/4, W/4, 4] format
    downsampled_image = downsampled_image.squeeze(0).permute(1, 2, 0)

    return downsampled_image


def load_data(path):
    raw = np.load(os.path.join(path))
    raw_img = raw["raw"]
    raw_max = raw["max_val"]
    # print(raw_img.dtype)
    raw_img = (raw_img / raw_max).astype(np.float32)
    # print(raw_img.shape)     # val_in : (512, 512, 4)   val_pred : (1024, 1024, 4)  对应的RGB (h*2, w*2, 3)
    return raw_img, raw_max


def load_data_folder(folder, hr_path, lr_path):
    raw_imgs = []
    raw_maxs = []
    for filename in os.listdir(folder):
        raw = np.load(os.path.join(folder, filename))
        raw_img = raw["raw"]
        raw_max = raw["max_val"]
        raw_img = (raw_img / raw_max).astype(np.float32)
        raw_imgs.append(raw_img)
        raw_maxs.append(raw_max)
        raw_img_downsampled = downsample_raw(torch.from_numpy(raw_img))
        raw_img_downsampled = raw_img_downsampled.detach().numpy()
        # # cv2.imshow('img1', raw_img)
        # # cv2.imshow('img2', raw_img_downsampled)
        np.save(os.path.join(hr_path, filename.replace('npz', 'npy')), raw_img)
        np.save(os.path.join(lr_path, filename.replace('npz', 'npy')), raw_img_downsampled)
    return raw_imgs, raw_maxs


def load_data_nonorm(path):
    raw = np.load(os.path.join(path))
    raw_img = raw["raw"]
    raw_max = raw["max_val"]
    return raw_img, raw_max

if __name__ == '__main__':
    # 注意，我们提供了清晰的原始图像。为 RAW 图像寻找有见地的退化pipeline是挑战的一部分。因此，从单个 RAW 可以生成数千个用于训练的低分辨率退化样本。
    # load_data_folder('F:\\Datasets\\DSLR\\train\\orig','F:/Datasets/DSLR/train/HR','F:\\Datasets\\DSLR\\train\\LR_bic2x' )
    for filename in os.listdir('F:\Datasets\DSLR\\train\orig'):
        raw, maxval = load_data_nonorm(os.path.join('F:\Datasets\DSLR\\train\orig', filename))
        # print(raw.dtype, maxval)
        if maxval > 16383:
            print(filename)
    print('Done!')
