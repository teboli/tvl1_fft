import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, fftpack
from skimage import data

from cpcr import utils, kernels


def edgetaper(y, k):
    y = utils.edgetaper(y.squeeze().cpu().numpy(), k.squeeze().cpu().numpy())
    y = torch.from_numpy(y).to(k.device).unsqueeze(0)
    return y


###########################################################
### Fast denominator computation with Fourier transform ###
###########################################################


def computeDenominator(y, k):
    sizey = y.shape[1:]
    otfk = kernels.psf2otf(k, sizey)

    # g1 = torch.zeros(1,2, device=k.device)
    # g1[0,0] = 1
    # g1[0,1] = -1
    g1 = torch.zeros(3, 3, device=k.device)
    g1[1,1] = 1
    g1[1,2] = -1
    g2 = g1.t()
    otfx = kernels.psf2otf(g1, sizey)
    otfy = kernels.psf2otf(g2, sizey)

    yfft = torch.rfft(y, 2, onesided=False)

    Nomin1 = utils.prod(utils.conj(otfk), yfft)

    Denom1 = utils.square_modulus(otfk)
    Denom2 = utils.square_modulus(otfx) + utils.square_modulus(otfy)

    return Nomin1, Denom1, Denom2


def computeDenominator3(y, z, k):
    sizey = y.shape[1:]
    otfk = psf2otf(k, sizey)
    
    g1 = torch.zeros(1,2, device=k.device)
    g1[0,0] = 1
    g1[0,1] = -1
    g2 = g1.t()
    otfx = kernels.psf2otf(g1, sizey)
    otfy = kernels.psf2otf(g2, sizey)
    
    yfft = torch.rfft(y, 2, onesided=False)
    zfft1 = torch.rfft(y, 2, onesided=False)
    zfft2 = torch.rfft(y, 2, onesided=False)
    
    Nomin1 = utils.prod(utils.conj(otfk), yfft)
    Nomin2 = utils.prod(utils.conj(otfx), zfft1) + utils.prod(utils.conj(otfy), zfft2)
    
    Denom1 = utils.square_modulus(otfk)
    Denom2 = utils.square_modulus(otfx) + utils.square_modulus(otfy)
    
    return Nomin1, Nomin2, Denom1, Denom2


############################################
### Direct FFT inversion for deblurring  ###
############################################


def FFT(y, k, beta):
    z = torch.zeros_like(y)
    N1, N2, D1, D2 = computeDenominator3(y, z, k)

    Fyout = (N1 + beta * N2) / (D1 + beta*D2)
    res = torch.irfft(Fyout, 2, signal_sizes=y.shape[1:], onesided=False)

    return res


##################################################################
### HQS inversion for deblurring with FFT least-squares solver ###
##################################################################


def diff(x, n, dim):
    assert dim in [1, 2]
    if dim == 1:
        out = x[:,n:,:] - x[:,:-n,:]
    else:
        out = x[...,n:] - x[...,:-n]
    return out


def HQS_FFT(y, k, lambd, n_iter, betas):
    # y is (1, c, h, w) tensor
    # k is (kh, kw) tensor
    if len(k.shape) > 2:
        k = k.squeeze()
    hks = k.shape[-1] // 2

    # betas = np.array([0, 1, 4, 4**2, 4**3, 4**4, 4**4, 4**5, 4**6, 4**7, 4**8]) * 1e-3 / 10 * 81

    hat_x = []
    # loop over color channels
    for c in range(y.shape[1]):
        y_in = y[:,c]
        yout = y_in.clone()
        # compute the denominator components
        N1, D1, D2 = computeDenominator(y_in, k)
        # initial gradient computation
        youtx = torch.cat([diff(yout, 1, 2), (yout[...,0] - yout[...,-1]).unsqueeze(-1)], -1)
        youty = torch.cat([diff(yout, 1, 1), (yout[:,0,:] - yout[:,-1,:]).unsqueeze(-2)], -2)

        # HQS main loop
        for pp in range(min(len(betas), n_iter)):
            beta = betas[pp]
            beta = max(1e-4, beta)
            gamma = beta / lambd
            D = D1 + beta * D2  # reweight the denominator with current HQS coupling weight

            # z update on current gradient estimates
            Wx = F.softshrink(youtx, lambd/beta)
            Wy = F.softshrink(youty, lambd/beta)

            Wxx = torch.cat([(Wx[..., -1] - Wx[..., 0]).unsqueeze(-1), -diff(Wx, 1, 2)], -1)
            Wxx = Wxx + torch.cat([(Wy[:, -1, :] - Wy[:, 0, :]).unsqueeze(-2), -diff(Wy, 1, 1)], -2)

            # x update with Fourier transform
            Fyout = (N1 + beta*torch.rfft(Wxx, 2, onesided=False)) / D
            yout = torch.irfft(Fyout, 2, signal_sizes=y_in.shape[1:], onesided=False)

            youtx = torch.cat([diff(yout,1,2), (yout[...,0] - yout[...,-1]).unsqueeze(-1)], -1)
            youty = torch.cat([diff(yout,1,1), (yout[:,0,:] - yout[:,-1,:]).unsqueeze(-2)], -2)
        hat_x.append(yout.unsqueeze(1))
    hat_x = torch.cat(hat_x, 1)
    return hat_x


def HQS_FFT_edgetaped(y, k, lambd, n_iter, betas):
    # y is (1, c, h, w) tensor
    # k is (kh, kw) tensor
    if len(k.shape) > 2:
        k = k.squeeze()
    hks = k.shape[-1] // 2

    hat_x = []

    # betas = np.array([0, 1, 4, 4**2, 4**3, 4**4, 4**5, 4**6, 4**7, 4**8]) * 1e-3 / 10 * 81

    for c in range(y.shape[1]):
        y_in = y[:,c]
        # edgetapping the image
        y_in = F.pad(y_in.unsqueeze(0), (hks, hks, hks, hks), mode='replicate')[0]
        y_in = edgetaper(y_in, k)
        y0 = y_in.clone()
        yout = y_in.clone()
        # compute the denominator
        N1, D1, D2 = computeDenominator(y_in, k)
        # initial guess of gradients
        youtx = torch.cat([diff(yout, 1, 2), (yout[...,0] - yout[...,-1]).unsqueeze(-1)], -1)
        youty = torch.cat([diff(yout, 1, 1), (yout[:,0,:] - yout[:,-1,:]).unsqueeze(-2)], -2)
        # mask for handling border artifacts
        mask = torch.zeros_like(y_in)
        mask[:,hks:-hks,:] = 1
        mask[...,hks:-hks] = 1

        ## HQS iteration
        for pp in range(min(len(betas), n_iter)):
            beta = betas[pp]
            beta = max(1e-4, beta)
            gamma = beta / lambd
            D = D1 + beta * D2
            
            # z-update
            Wx = F.softshrink(youtx, lambd/beta)
            Wy = F.softshrink(youty, lambd/beta)

            Wxx = torch.cat([(Wx[...,-1] - Wx[...,0]).unsqueeze(-1), -diff(Wx,1,2)], -1)
            Wxx = Wxx + torch.cat([(Wy[:,-1,:] - Wy[:,0,:]).unsqueeze(-2), -diff(Wy,1,1)], -2)

            # 
            if pp > 0:
                y_in = y0
                otfk = utils.psf2otf(k, y_in.shape[1:])
                x_in = utils.prod(torch.rfft(yout, 2, onesided=False), otfk)
                x_in = torch.irfft(x_in, 2, signal_sizes=y_in.shape[1:], onesided=False)
                y_in = y_in * mask + (1-mask) * x_in

            N1, D1, D2 = computeDenominator(y_in, k)

            Fyout = (N1 + beta*torch.rfft(Wxx,2,onesided=False)) / D
            yout = torch.irfft(Fyout, 2, signal_sizes=y_in.shape[1:], onesided=False)

            youtx = torch.cat([diff(yout,1,2), (yout[...,0] - yout[...,-1]).unsqueeze(-1)], -1)
            youty = torch.cat([diff(yout,1,1), (yout[:,0,:] - yout[:,-1,:]).unsqueeze(-2)], -2)

        hat_x.append(yout.unsqueeze(1))
    hat_x = torch.cat(hat_x, 1)
    return hat_x[..., hks:-hks, hks:-hks]

if __name__ == '__main__':
    import pdb
    import time

    y = torch.rand(1, 500, 375).cuda()
    k = torch.rand(35, 35).cuda()

    lambd = 0.05
    n_iter = 10

    start = time.time()
    hat_x = HQS_FFT_edgetaped(y, k, lambd, n_iter)
    
    print(time.time()-start)
    print(hat_x.device)
