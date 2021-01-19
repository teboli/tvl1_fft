import torch
import torch.nn.functional as F

import numpy as np
from scipy import linalg
import os

import random

import utils


def psf2otf(psf, img_shape):
    # build padded array
    psf_shape = psf.shape
    h_pad = (img_shape[0] - psf_shape[0]) // 2
    w_pad = (img_shape[1] - psf_shape[1]) // 2
    psf = psf.unsqueeze(0).unsqueeze(0)
    if h_pad > 0:
        psf = F.pad(psf, (w_pad, w_pad, h_pad, h_pad))
    if psf.shape[-1] < img_shape[-1]:
        psf = F.pad(psf, (0, 1, 0, 0))
    if psf.shape[-2] < img_shape[-2]:
        psf = F.pad(psf, (0, 0, 0, 1))
    psf = psf.squeeze(0)
    
    # circular shift
    for axis, axis_size in enumerate(img_shape):
        psf = torch.roll(psf, -int(axis_size) // 2+1, axis+1)

    # compute OTF
    otf = torch.rfft(psf, 2, onesided=False)
    return otf


def otf2psf(otf, psf_shape):
    # compute PSF
    psf = torch.irfft(otf, 2, onesided=False)[0]
    
    # circular shift
    otf_shape = otf.shape[1:]
    for axis in [1, 0]:
        axis_size = otf_shape[axis]
        psf = torch.roll(psf, int(axis_size) // 2, axis)
    
    # build cropped kernel
    h_pad = (otf_shape[0] - psf_shape[0]) // 2
    w_pad = (otf_shape[1] - psf_shape[1]) // 2
    if h_pad > 0:
        psf = psf[h_pad:-h_pad, w_pad:-w_pad]
    
    return psf
