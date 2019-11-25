#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2019 qiang.zhou <theodoruszq@gmail.com>

"""

"""

import torch.nn as nn
import torch
from torch.nn import functional as F
import os, sys, functools


from inplace_ABN import InPlaceABN, InPlaceABNSync
BatchNorm2d = functools.partial(InPlaceABNSync, activation='none')
#BatchNorm2d = nn.BatchNorm2d

#import warnings
#warnings.filterwarnings('ignore')


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=False)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual      
        out = self.relu(out)

        return out

class ResidualBlock(nn.Module):
    def __init__(self, v):
        super(ResidualBlock, self).__init__()
        self.res = nn.Sequential(
            nn.ReLU(inplace=False),
            BatchNorm2d(v),
            nn.Conv2d(v, v, kernel_size=3, padding=1, bias=True), 
            nn.ReLU(inplace=False),
            BatchNorm2d(v),
            nn.Conv2d(v, v, kernel_size=3, padding=1, bias=True))

    def forward(self, x):
        x = x + self.res(x)
        return x

class GlobalConvolutionBlock(nn.Module):
    def __init__(self, in_dim, out_dim=256, k=7):
        super(GlobalConvolutionBlock, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=(1,k), padding=(0, k//2), bias=True), 
            nn.Conv2d(out_dim, out_dim, kernel_size=(k,1), padding=(k//2, 0), bias=True))

        self.branch2 = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=(k,1), padding=(0, k//2), bias=True), 
            nn.Conv2d(out_dim, out_dim, kernel_size=(1,k), padding=(k//2, 0), bias=True))

        self.RB = ResidualBlock(out_dim)

    def forward(self, x):
        out = self.branch1(x) + self.branch2(x)
        out = self.RB(out)
        return out


class RefinementModule(nn.Module):
    def __init__(self, in_dim, out_dim=256):
        super(RefinementModule, self).__init__()
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1, bias=True)
        self.RB1 = ResidualBlock(out_dim)
        self.RB2 = ResidualBlock(out_dim)

    def forward(self, x_top, x_low):
        _, _, h, w = x_low.size()

        x_top = F.interpolate(x_top, size=(h, w), mode='bilinear', align_corners=False)
        x_low = self.RB1(self.conv(x_low))
        x = x_top + x_low
        x = self.RB2(x)
        return x

class MultiLevelFeatureAgger(nn.Module):
    def __init__(self):
        super(MultiLevelFeatureAgger, self).__init__()
        self.conv_s1 = nn.Sequential(
                        nn.Conv2d(257, 64, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(inplace=False),
                        BatchNorm2d(64))
        self.conv_s2 = nn.Sequential(
                        nn.Conv2d(257, 64, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(inplace=False),
                        BatchNorm2d(64))
        self.conv_s3 = nn.Sequential(
                        nn.Conv2d(257, 64, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(inplace=False),
                        BatchNorm2d(64))
        self.conv_s4 = nn.Sequential(
                        nn.Conv2d(257, 64, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(inplace=False),
                        BatchNorm2d(64))
        self.agg = ResidualBlock(256)

    def forward(self, ref_feat):
        s1, s2, s3, s4 = ref_feat           # 1/4 -> 1/32
        _, _, h, w = s4.size()
        s1 = self.conv_s1(s1)
        s2 = self.conv_s2(s2)
        s3 = self.conv_s2(s3)
        s4 = self.conv_s2(s4)
        s1 = F.interpolate(s1, size=(h, w), mode='bilinear', align_corners=False)
        s2 = F.interpolate(s2, size=(h, w), mode='bilinear', align_corners=False)
        s3 = F.interpolate(s3, size=(h, w), mode='bilinear', align_corners=False)
        s4 = F.interpolate(s4, size=(h, w), mode='bilinear', align_corners=False)
        s = torch.cat([s1, s2, s3, s4], dim=1)
        s = self.agg(s)
        return s

class MultiLevelFeatureRect(nn.Module):
    def __init__(self):
        super(MultiLevelFeatureRect, self).__init__()
        self.conv_s1 = nn.Sequential(
                        nn.Conv2d(257, 256, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(inplace=False),
                        BatchNorm2d(256))
        self.conv_s2 = nn.Sequential(
                        nn.Conv2d(257, 256, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(inplace=False),
                        BatchNorm2d(256))
        self.conv_s3 = nn.Sequential(
                        nn.Conv2d(257, 256, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(inplace=False),
                        BatchNorm2d(256))
        self.conv_s4 = nn.Sequential(
                        nn.Conv2d(257, 256, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(inplace=False),
                        BatchNorm2d(256))

    def forward(self, ref_feat):
        s1, s2, s3, s4 = ref_feat           # 1/4 -> 1/32
        _, _, h, w = s4.size()
        s1 = self.conv_s1(s1)
        s2 = self.conv_s2(s2)
        s3 = self.conv_s2(s3)
        s4 = self.conv_s2(s4)
        return [s1, s2, s3, s4]

class DRSN(nn.Module):
    def __init__(self, out_size=(256, 256), model_path=None):
        super(DRSN, self).__init__()

        self.GCN = GlobalConvolutionBlock(512)
        self.RefAgg = MultiLevelFeatureAgger()
        self.CurRect = MultiLevelFeatureRect()
        self.RM1 = RefinementModule(256)
        self.RM2 = RefinementModule(256)
        self.RM3 = RefinementModule(256)
        self.classifier = nn.Sequential(
                            nn.Conv2d(256, 256, kernel_size=3, padding=1),
                            BatchNorm2d(256),
                            nn.ReLU(inplace=False),
                            nn.Dropout2d(0.1),
                            nn.Conv2d(256, 2, kernel_size=1, stride=1, padding=0))

        self.out_size = out_size
        self.init_weights(model_path)

    """
    Args:
        :ref_feat: reference features, it has four scales and concated with its masks.
                   All of them have 257 channels, and their scales are [1/4, 1/8, 1/16, 1/32]
        :x_feat: current features, it has the same setting with ref_feat.

        Note only [1/4, 1/8, 1/16] will be refined.
    Return:
        :out: upsampled masks produced by DRSN.
    """
    def forward(self, ref_feat, x_feat):
        assert len(ref_feat) == len(x_feat)

        ref_feat_agg = self.RefAgg(ref_feat)
        x_feat = self.CurRect(x_feat)

        concat_feat = torch.cat((ref_feat_agg, x_feat[-1]), dim=1)
        gcn_feat = self.GCN(concat_feat)
        refine_feat = self.RM1(gcn_feat, x_feat[-2])
        refine_feat = self.RM2(refine_feat, x_feat[-3])
        refine_feat = self.RM3(refine_feat, x_feat[-4])
        
        out = F.interpolate(self.classifier(refine_feat), size=self.out_size, mode='bilinear', align_corners=False)
        return out

    def set_bn_status(self, freeze=True):
        for m in self.modules():
            if isinstance(m, InPlaceABNSync):
                if freeze is True:
                    m.eval()
                else:
                    m.train()
    
    def init_weights(self, model_path=None):
        def init_func(m):
            import math
            #print (m)
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2./n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        if model_path is None:
            self.apply(init_func)
        else:
            raise NotImplementedError


class CrossEntropyLoss_Softmax(nn.Module):
    def __init__(self, reduction='sum', ignore_index=255):
        super(CrossEntropyLoss_Softmax, self).__init__()
        self.loss_func = torch.nn.CrossEntropyLoss(reduction=reduction,
                                                   ignore_index=ignore_index)

    def forward(self, preds, target):
        loss = self.loss_func(preds, target)
        return loss


if __name__ == "__main__":
    drsn = DRSN().cuda()
    #drsn.init_weights()
    #exit()

    aa = torch.ones((1, 257, 64, 64)).cuda()
    ab = torch.ones((1, 257, 32, 32)).cuda()
    ac = torch.ones((1, 257, 16, 16)).cuda()
    ad = torch.ones((1, 257,  8,  8)).cuda()
    ref = [aa, ab, ac, ad]
    x = [aa, ab, ac, ad]
    drsn(ref, x)
    drsn.backward()
    exit()

    c = CrossEntropyLoss_Softmax()
    p = torch.ones((1, 2, 256, 256)).type(torch.cuda.FloatTensor)
    t = torch.zeros((1, 256, 256)).type(torch.cuda.LongTensor)
    c(p, t)


        

        
        


