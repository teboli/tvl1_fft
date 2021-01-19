import os

import torch
import torch.nn.functional as F
import numpy as np
from scipy.signal import fftconvolve
from torchvision import transforms

from skimage import transform

import pdb
import matplotlib.pyplot as plt


#############################
#### COnversion routines ####
#############################

def to_tensor(array):
    # from hxwxc to cxhcw
    if type(array) == torch.Tensor:
        return array
    else:
        tensor = torch.from_numpy(array)
        if len(tensor.shape) == 2:
            tensor = tensor.unsqueeze(0)
        else:
            tensor = tensor.permute(2, 0, 1)
        return tensor.float()


def to_array(tensor):
    if type(tensor) == np.ndarray:
        return tensor
    else:
        tensor = tensor.squeeze()
        if len(tensor.shape) == 3:
            tensor = tensor.permute(1, 2, 0)
        array = tensor.detach().cpu().numpy()
        return array


def to_float(img):
    img = img_as_float(img)
    img = img.astype(np.float32)
    return img


def to_uint(img):
    img = img_as_float(img)
    img = (255*img).astype(np.uint8)
    return img


############################
### Edgetapping routines ###
############################


# from kruse et al. 17
def pad_for_kernel(img, kernel, mode):
    p = [(d-1)//2 for d in kernel.shape]
    padding = [p, p] + (img.ndim-2)*[(0, 0)]
    return np.pad(img, padding, mode)


# from kruse et al. 17
def crop_for_kernel(img, kernel):
    p = [(d-1)//2 for d in kernel.shape]
    r = [slice(p[0], -p[0]), slice(p[1], -p[1])] + (img.ndim-2)*[slice(None)]
    return img[r]

# from kruse et al. 17
def pad_for_kernel_v2(img, kernel, mode):
    p = [(int(1.8*d)-1)//2 for d in kernel.shape]
    padding = [p, p] + (img.ndim-2)*[(0, 0)]
    return np.pad(img, padding, mode)


# from kruse et al. 17
def crop_for_kernel_v2(img, kernel):
    p = [(int(1.8*d)-1)//2 for d in kernel.shape]
    r = [slice(p[0], -p[0]), slice(p[1], -p[1])] + (img.ndim-2)*[slice(None)]
    return img[r]


# from kruse et al. 17
def edgetaper_alpha(kernel, img_shape):
    v = []
    for i in range(2):
        z = np.fft.fft(np.sum(kernel, 1-i), img_shape[i]-1)
        z = np.real(np.fft.ifft(np.square(np.abs(z)))).astype(np.float32)
        z = np.concatenate([z, z[0:1]], 0)
        v.append(1 - z/np.max(z))
    return np.outer(*v)


# from kruse et al. 17
def edgetaper(img, kernel, n_tapers=3):
    alpha = edgetaper_alpha(kernel, img.shape[0:2])
    _kernel = kernel
    if 3 == img.ndim:
        kernel = kernel[..., np.newaxis]
        alpha  = alpha[..., np.newaxis]
    for i in range(n_tapers):
        blurred = fftconvolve(pad_for_kernel(img, _kernel,'wrap'), kernel, mode='valid')
        img = alpha*img + (1-alpha)*blurred
    return img


#############################################
### Complex number operations for pytorch ###
#############################################


def conj(A):
    B = A.clone()
    B[...,-1] *= -1
    return B


def prod(A, B):
    AB1 = A[...,0]*B[...,0] - A[...,1]*B[...,1]
    AB2 = A[...,0]*B[...,1] + A[...,1]*B[...,0]
    return torch.cat([AB1.unsqueeze(-1), AB2.unsqueeze(-1)], dim=-1)


def square_modulus(A):
    p = prod(conj(A), A)
    p = p.sum(-1, keepdim=True)
    return p


def div(A, B):
    mod2_B = square_modulus(B)
    prod_AB = prod(A, conj(B))
    return prod_AB / mod2_B
