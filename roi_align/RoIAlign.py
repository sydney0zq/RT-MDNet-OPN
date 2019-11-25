#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2019 qiang.zhou <theodoruszq@gmail.com>

from torch.autograd import Function
from csrc._C import roi_align_forward, roi_align_backward
import torch.nn as nn

class RoIAlign(nn.Module):
    def __init__(self, aligned_height, aligned_width, spatial_scale):
        super(RoIAlign, self).__init__()
        self.aligned_width = int(aligned_width)
        self.aligned_height = int(aligned_height)
        self.spatial_scale = float(spatial_scale)

    def forward(self, features, rois):
        return RoIAlignFunction(self.aligned_height, self.aligned_width, self.spatial_scale)(features, rois)

class RoIAlignFunction(Function):
    def __init__(self, aligned_height, aligned_width, spatial_scale):
        self.aligned_width = int(aligned_width)
        self.aligned_height = int(aligned_height)
        self.spatial_scale = float(spatial_scale)
        self.rois = None
        self.feature_size = None

    def forward(self, features, rois):
        self.rois = rois

        batch_size, num_channels, data_height, data_width = features.size()
        num_rois = rois.size(0)

        output = roi_align_forward(features, rois, self.spatial_scale, self.aligned_height, self.aligned_width, 2)
        return output

    def backward(self, grad_output):
        assert(self.feature_size is not None and grad_output.is_cuda)
        batch_size, num_channels, data_height, data_width = self.feature_size

        grad_input = roi_align_backward(grad_output, self.rois, self.spatial_scale, self.aligned_height, self.aligned_width, batch_size,
                    num_channels, data_height, data_width, 2)
        return grad_input, None


if __name__ == "__main__":
    import torch
    import numpy as np
    aa = torch.ones((1, 3, 100, 100))
    print (aa.size())
    print (torch.Tensor([[10, 10, 20, 20]]).size())
    roialign = RoIAlign(10, 10, 1)
    aa = aa.cuda()
    #roialign = roialign.cuda()
    #roialign(aa, torch.Tensor([[10, 10, 20, 20]]).cuda())

    roif = RoIAlignFunction(10, 10, 1)

    loss = roif(aa, torch.Tensor(([10, 10, 20, 20])).cuda())
    #loss.backward()

        
