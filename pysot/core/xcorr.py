# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import math

import torch
import torch.nn.functional as F


def xcorr_slow(x, kernel):
    """for loop to calculate cross correlation, slow version
    """
    batch = x.size()[0]
    out = []
    for i in range(batch):
        px = x[i]
        pk = kernel[i]
        px = px.view(1, -1, px.size()[1], px.size()[2])
        pk = pk.view(1, -1, pk.size()[1], pk.size()[2])
        po = F.conv2d(px, pk)
        out.append(po)
    out = torch.cat(out, 0)
    return out

def xcorr_fast(x, kernel):
    """group conv2d to calculate cross correlation, fast version
    """
    batch = kernel.size()[0]
    pk = kernel.view(-1, x.size()[1], kernel.size()[2], kernel.size()[3])
    px = x.view(1, -1, x.size()[2], x.size()[3])
    po = F.conv2d(px, pk, groups=batch)
    po = po.view(batch, -1, po.size()[2], po.size()[3])
    return po

def xcorr_depthwise(x, kernel):
    """depthwise cross correlation
    """
    batch = kernel.size(0)
    channel = kernel.size(1)
    x = x.view(1, batch*channel, x.size(2), x.size(3))
    kernel = kernel.view(batch*channel, 1, kernel.size(2), kernel.size(3))
    out = F.conv2d(x, kernel, groups=batch*channel)
    out = out.view(batch, channel, out.size(2), out.size(3))

    return out

def xcorr_pixelwise(x,kernel): #z=kernel
    """Pixel-wise correlation (implementation by matrix multiplication)
    The speed is faster because the computation is vectorized"""
    b, c, h, w = x.size()
    kernel_mat = kernel.view((b, c, -1)).transpose(1, 2)  # (b, hz * wz, c)
    x_mat = x.view((b, c, -1))  # (b, c, hx * wx)
    return torch.matmul(kernel_mat, x_mat).view((b, -1, h, w))  # (b, hz * wz, hx * wx) --> (b, hz * wz, hx, wx)

def pg_xcorr(search, kernel):
    # b, c, h, w = search.shape
    # ker1 = kernel.reshape(b, c, -1)
    # ker2 = ker1.transpose(1, 2)
    # feat = search.reshape(b, c, -1)
    # S1 = torch.matmul(ker2, feat)
    # S2 = torch.matmul(ker1, S1)
    # corr = S2.reshape(*S2.shape[:2], h, w)

    b, c, h, w = search.shape
    ker1 = kernel.reshape(b, c, -1)
    ker2 = ker1.transpose(1, 2)
    S1 = xcorr_pixelwise(search, ker1)
    S2 = xcorr_pixelwise(S1,ker2)
    corr = torch.cat([S1, S2], 1)
    return corr


def non_local_xcorr(fm, fq): # z, x
    # TODO: SiamPW-RBO中的PW卷积
    B, C, h, w = fm.shape

    B, C, H, W = fq.shape
    # print("fm shape:",fm.shape)
    # print("fq shape:",fq.shape)
    fm0 = fm.clone()
    fq0 = fq.clone()

    fm = fm.contiguous().view(B, C, h * w)  # B, C, hw

    fm = fm.permute(0, 2, 1)  # B, hw, C

    fq = fq.contiguous().view(B, C, H * W)  # B, C, HW

    # print("fm shape:",fm.shape)
    # print("fq shape:",fq.shape)
    similar = torch.matmul(fm, fq) / math.sqrt(C)  # B, hw, HW
    # print("w shape:",similar.shape)

    similar = torch.softmax(similar, dim=1)  # B, hw, HW

    fm1 = fm0.view(B, C, h * w)  # B, C, hw
    mem_info = torch.matmul(fm1, similar)  # (B, C, hw) x (B, hw, HW) = (B, C, HW)
    mem_info = mem_info.view(B, C, H, W)

    y = torch.cat([mem_info, fq0], dim=1)
    return y