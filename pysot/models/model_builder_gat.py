# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from pysot.core.config import cfg
from pysot.models.loss_car import make_siamcar_loss_evaluator, rank_cls_loss
from pysot.models.backbone import get_backbone
from pysot.models.head.car_head import CARHead
from ..core.xcorr import xcorr_pixelwise, pg_xcorr
from ..utils.location_grid import compute_locations
from ..utils.xcorr import xcorr_depthwise

class CAModule(nn.Module):
    """Channel attention module"""

    def __init__(self, channels=256, reduction=1):
        super(CAModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1,
                             padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1,
                             padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


class PixelwiseXCorr(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(PixelwiseXCorr, self).__init__()

        self.CA_layer = CAModule(channels=256)

    def forward(self, kernel, search):
        #feature = xcorr_pixelwise(search, kernel)

        feature = pg_xcorr(search, kernel)
        corr = self.CA_layer(feature)

        return corr
#
#
# class Graph_Attention_Union(nn.Module):
#     def __init__(self, in_channel, out_channel):
#         super(Graph_Attention_Union, self).__init__()
#
#         # search region nodes linear transformation
#         self.support = nn.Conv2d(in_channel, in_channel, 1, 1)
#
#           # target template nodes linear transformation
#         self.query = nn.Conv2d(in_channel, in_channel, 1, 1)
#         self.xcorr_depthwise = xcorr_depthwise
#         # linear transformation for message passing
#         self.g = nn.Sequential(
#             nn.Conv2d(in_channel, in_channel, 1, 1),
#             nn.BatchNorm2d(in_channel),
#             nn.ReLU(inplace=True),
#         )
#
#         # aggregated feature
#         self.fi = nn.Sequential(
#             nn.Conv2d(in_channel*2, out_channel, 1, 1),
#             nn.BatchNorm2d(out_channel),
#             nn.ReLU(inplace=True),
#         )
#
#     def forward(self, zf, xf):
#         # linear transformation
#         #对应等式2，对搜索图像节点、模版图像节点做了一个线性变换（1*1卷积）
#         xf_trans = self.query(xf)
#         zf_trans = self.support(zf)
#
#         # linear transformation for message passing
#         # 对应等式4和5，同样也是先做线性变换
#         xf_g = self.g(xf)  #[b,c,w,h]
#         zf_g = self.g(zf)  #[b,c,w,h]
#
#         # calculate similarity 计算相似度
#         shape_x = xf_trans.shape #[b,c,w,h]
#         shape_z = zf_trans.shape #[b,c,w,h]
#
#         zf_trans_plain = zf_trans.view(-1, shape_z[1], shape_z[2] * shape_z[3])  #[b,c,w*h]
#         zf_g_plain = zf_g.view(-1, shape_z[1], shape_z[2] * shape_z[3]).permute(0, 2, 1) # [b,c,w*h]->[b, w*h,c]
#         # 矩阵转置
#         xf_trans_plain = xf_trans.view(-1, shape_x[1], shape_x[2] * shape_x[3]).permute(0, 2, 1) #[b, w*h,c]
#
#         # 论文是将一个[1,1,c]当作一个节点向量进行运算
#         similar = torch.matmul(xf_trans_plain, zf_trans_plain) # eij=x[b,w*h,c]*z[b,c,w*h]
#         similar = F.softmax(similar, dim=2) #softmax,求得aij
#
#         embedding = torch.matmul(similar, zf_g_plain).permute(0, 2, 1)
#         embedding = embedding.view(-1, shape_x[1], shape_x[2], shape_x[3]) # emdedding得到Vi
#
#         # aggregated feature
#         # output = torch.cat([embedding, xf_g], 1)
#         # output = self.fi(output)
#
#         #TODO: 把融合再降低通道改为点乘,降低运算量,提高10fps
#         features = torch.mul(embedding,xf_g)
#
#         return features #返回响应图


class ModelBuilder(nn.Module):
    def __init__(self):
        super(ModelBuilder, self).__init__()

        # build backbone
        self.backbone = get_backbone(cfg.BACKBONE.TYPE,
                                     **cfg.BACKBONE.KWARGS)

        # build car head
        self.car_head = CARHead(cfg, 256)

        # build response map,使用GAT提供的GAT模块
        #self.attention = Graph_Attention_Union(256, 256)

        # 使用PW卷积
        '''
        self.attention = PixelwiseXCorr(256,256)
        # 简单粗暴提高通道数
        #self.up = nn.ConvTranspose2d(169, 256, 1, 1)

        # 卷积-> norm -> ReLU
        self.up = nn.Sequential(
                    nn.Conv2d(169 , 256, 1, 1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True),
                )
        '''
        # build loss
        self.loss_evaluator = make_siamcar_loss_evaluator(cfg)
        #self.rank_cls_loss = rank_cls_loss()
        #self.cen_iou_loss = cen_iou_loss()




    def template(self, z, roi):
        zf = self.backbone(z, roi)
        self.zf = zf



# 跟踪过程
    def track(self, x):
        xf = self.backbone(x)

        #features = self.attention(self.zf, xf)
        # # TODO: 在线更新写在这里,需要更新的是模版的特征
        # features = self.up(features)
        #cls, loc, cen = self.car_head(features)
        cls, loc, cen = self.car_head(self.zf, xf)

        return {
                'cls': cls,
                'loc': loc,
                'cen': cen
               }

    def log_softmax(self, cls):
        b, a2, h, w = cls.size()
        cls = cls.view(b, 2, a2//2, h, w)
        cls = cls.permute(0, 2, 3, 4, 1).contiguous()
        cls = F.log_softmax(cls, dim=4)
        return cls



    def forward(self, data):
        """ only used in training
        """
        # target_box和label_loc的区别：前者是label，后者是mask
        template = data['template'].cuda()
        search = data['search'].cuda()
        label_cls = data['label_cls'].cuda()
        label_loc = data['bbox'].cuda()
        target_box = data['target_box'].cuda()
        neg = data['neg'].cuda()

        # get feature
        zf = self.backbone(template, target_box)
        xf = self.backbone(search)

        # features = self.attention(zf, xf)
        # features = self.up(features)
        cls, loc, cen = self.car_head(zf, xf)
        locations = compute_locations(cls, cfg.TRACK.STRIDE, cfg.TRACK.OFFSET)
        cls = self.log_softmax(cls)
        cls_loss, loc_loss, cen_loss, Cls_Rank_loss =  self.loss_evaluator(
            locations,
            cls,
            loc,
            cen, label_cls, label_loc, neg
        )
        # cen_iou_loss_1 = self.cen_iou_loss(cen ,label_cls,loc,label_cls)
        outputs = {}
        # outputs['total_loss'] = cfg.TRAIN.CLS_WEIGHT * cls_loss + \
        #     cfg.TRAIN.LOC_WEIGHT * loc_loss + cfg.TRAIN.CEN_WEIGHT * cen_loss

        outputs['total_loss'] = (cfg.TRAIN.CLS_WEIGHT * cls_loss + \
           cfg.TRAIN.LOC_WEIGHT * loc_loss + cfg.TRAIN.CEN_WEIGHT * cen_loss + 0.5 * Cls_Rank_loss ).reshape(1)

        # print((label_cls == 1).sum())
        outputs['cls_loss'] = cls_loss.reshape(1)
        outputs['loc_loss'] = loc_loss.reshape(1)
        outputs['cen_loss'] = cen_loss.reshape(1)
        outputs['Cls_Rank_loss'] = Cls_Rank_loss.reshape(1)
        # outputs['Cen_Rank_loss'] = Cen_Rank_loss.reshape(1)

        return outputs

