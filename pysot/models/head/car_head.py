import torch
from torch import nn
import math

from pysot.core.xcorr import xcorr_pixelwise, pg_xcorr


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
        self.g = nn.Sequential(
            nn.Conv2d(425, 256, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.CA_layer = CAModule(channels=256)

    def forward(self, kernel, search):
        # feature = xcorr_pixelwise(search, kernel)  #
        # 没有分别对kernel和search进行操作
        feature = pg_xcorr(search, kernel)
        # concat原特征并降维
        feature2 = self.g(feature);
        corr = self.CA_layer(feature2)

        return corr
class CARHead(torch.nn.Module):
    def __init__(self, cfg, in_channels):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super(CARHead, self).__init__()
        # TODO: Implement the sigmoid version first.
        num_classes = cfg.TRAIN.NUM_CLASSES

        cls_tower = []
        bbox_tower = []
        for i in range(cfg.TRAIN.NUM_CONVS):
            cls_tower.append(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1
                )
            )

            #cls_tower.append(nn.GroupNorm(32, in_channels))
            cls_tower.append(nn.GroupNorm(32, in_channels))
            cls_tower.append(nn.ReLU())
            bbox_tower.append(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1
                )
            )
            bbox_tower.append(nn.GroupNorm(32, in_channels))
            bbox_tower.append(nn.ReLU())

        self.add_module('cls_tower', nn.Sequential(*cls_tower))
        self.add_module('bbox_tower', nn.Sequential(*bbox_tower))
        self.cls_logits = nn.Conv2d(
            in_channels, num_classes, kernel_size=3, stride=1,
            padding=1
        )
        self.bbox_pred = nn.Conv2d(
            in_channels, 4, kernel_size=3, stride=1,
            padding=1
        )
        self.centerness = nn.Conv2d(
            in_channels, 1, kernel_size=3, stride=1,
            padding=1
        )
        self.cls_pw = PixelwiseXCorr(256,256)
        # self.cls_up = nn.Sequential(
        #             nn.Conv2d(169 , 256, 1, 1),
        #             nn.BatchNorm2d(256),
        #             nn.ReLU(inplace=True),
        #         )
        self.reg_pw = PixelwiseXCorr(256, 256)
        # self.reg_up = nn.Sequential(
        #     nn.Conv2d(169, 256, 1, 1),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(inplace=True),
        # )
        # initialization
        for modules in [self.cls_tower, self.bbox_tower,
                        self.cls_logits, self.bbox_pred,
                        self.centerness]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)

        # initialize the bias for focal loss,
        prior_prob = cfg.TRAIN.PRIOR_PROB
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_logits.bias, bias_value)

    def forward(self, z_f, x_f):
        # TODO: 把centerness移动到reg模块
        # cls_tower = self.cls_tower(x)
        # logits = self.cls_logits(cls_tower)
        # centerness = self.centerness(cls_tower)
        # bbox_reg = torch.exp(self.bbox_pred(self.bbox_tower(x)))
        #TODO :   修改后

        # cls_tower = self.cls_tower(x)
        # logits = self.cls_logits(cls_tower)
        # bbox_tower = self.bbox_tower(x)
        # centerness = self.centerness(bbox_tower)
        # bbox_reg = torch.exp(self.bbox_pred(bbox_tower))
        # return logits, bbox_reg, centerness

        # TODO：使用两个pw分别操作
        x_cls = self.cls_pw(z_f, x_f)
#        x = self.cls_up(x_cls)
        x_reg = self.reg_pw(z_f, x_f)
#        x1 = self.reg_up(x_reg)

        #cls分支
        cls_tower = self.cls_tower(x_cls)
        logits = self.cls_logits(cls_tower)
        # reg分支
        bbox_tower = self.bbox_tower(x_reg)
        centerness = self.centerness(bbox_tower)
        bbox_reg = torch.exp(self.bbox_pred(bbox_tower))
        return logits, bbox_reg, centerness


class Scale(nn.Module):
    def __init__(self, init_value=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale

