import numpy as np
from matplotlib import pyplot as plt
import cv2
import random

from scipy.io import loadmat
from scipy import ndimage
from scipy.signal import convolve2d
import hdf5storage

import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy import signal
import kornia

from .imutils import postprocess_raw, demosaic, save_rgb, plot_all


def augment_kernel(kernel, mode=0):
    '''Kai Zhang (github: https://github.com/cszn)
    Rotate kernels (or images)
    '''
    if mode == 0:
        return kernel
    elif mode == 1:
        return np.flipud(np.rot90(kernel))
    elif mode == 2:
        return np.flipud(kernel)
    elif mode == 3:
        return np.rot90(kernel, k=3)
    elif mode == 4:
        return np.flipud(np.rot90(kernel, k=2))
    elif mode == 5:
        return np.rot90(kernel)
    elif mode == 6:
        return np.rot90(kernel, k=2)
    elif mode == 7:
        return np.flipud(np.rot90(kernel, k=3))


def apply_custom_filter(inp_img, kernel_):
    return kornia.filters.filter2d(inp_img, kernel_, normalized=True)


def generate_gkernel(ker_sz=None, sigma=None):
    gkern1 = signal.gaussian(ker_sz, std=sigma[0]).reshape(ker_sz, 1)
    gkern2 = signal.gaussian(ker_sz, std=sigma[1]).reshape(ker_sz, 1)
    gkern = np.outer(gkern1, gkern2)
    return gkern


def apply_gkernel(inp_img, ker_sz=5, ksigma_vals=[.05 + i for i in range(5)]):
    """
    Apply uniform gaussian kernel of sizes between 5 and 11.
    """
    # sample for variance
    sigma_val1 = ksigma_vals[np.random.randint(len(ksigma_vals))]
    sigma_val2 = ksigma_vals[np.random.randint(len(ksigma_vals))]
    sigma = (sigma_val1, sigma_val2)

    kernel = generate_gkernel(ker_sz, sigma)
    tkernel = torch.from_numpy(kernel.copy()).view(1, ker_sz, ker_sz).type(torch.FloatTensor)
    blurry = apply_custom_filter(inp_img, tkernel)
    return torch.clamp(blurry, 0., 1.), kernel


def apply_psf(inp_img, kernels):
    """
    Apply PSF
    """
    idx = np.random.choice(np.arange(11),
                           p=[0.15, 0.20, 0.20, 0.0075, 0.0075, 0.175, 0.175, 0.05, 0.0075, 0.0075, 0.02])
    kernel = kernels[idx].astype(np.float64)
    kernel = augment_kernel(kernel, mode=random.randint(0, 7))
    ker_sz = 25
    tkernel = torch.from_numpy(kernel.copy()).view(1, ker_sz, ker_sz).type(torch.FloatTensor)
    blurry = apply_custom_filter(inp_img, tkernel)
    return torch.clamp(blurry, 0., 1.), kernel


def add_blur(inp_img, kernels, plot=False, gkern_szs=[3, 5, 7, 9]):
    # sample for kernel size
    ker_sz = gkern_szs[np.random.randint(len(gkern_szs))]
    use_gkernel = random.random() > 0.5
    kernel_type = ''

    if use_gkernel:
        kernel_type = f'gaussian_{ker_sz}'
        # blurry, kernel = apply_gkernel(inp_img.unsqueeze(0), ker_sz=ker_sz)
        blurry, kernel = apply_gkernel(inp_img, ker_sz=ker_sz)
    else:
        kernel_type = f'psf'
        # blurry, kernel = apply_psf(inp_img.unsqueeze(0), kernels)
        blurry, kernel = apply_psf(inp_img, kernels)   # for batch

    # if plot:
    #     kernelid = np.random.randint(999999)
    #     print ('Kernel', kernelid, kernel.shape, kernel_type)
    #     save_rgb((kernel*255).astype(np.float32), f'kernel_{kernelid}.png')

    return blurry


if __name__ == "__main__":
    pass
