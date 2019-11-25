#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2019 qiang.zhou <theodoruszq@gmail.com>


import cv2
import numpy as np

""" Put boxes on image, image and color should keep consist in channel dim.
    :image: Numpy array image in HxWxC.
    :box: Nx4 contains N boxes, it should in cross format, (minx, miny, maxx, maxy).
    :thickness: 
"""
def overlay_box(image, box, color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA):
    image = np.copy(image)
    if len(box.shape) == 1:
        box = np.array([box])
    for i_box in box:
        minx, miny, maxx, maxy = i_box
        if int(cv2.__version__[0]) == 4:
            image = cv2.rectangle(image, (int(minx), int(miny)), (int(maxx), int(maxy)), \
                    color=color, thickness=thickness, lineType=lineType)
        else:
            print ("cv2 version is lower than 4.0, be careful here...")
            cv2.rectangle(image, (int(minx), int(miny)), (int(maxx), int(maxy)), \
                    color=color, thickness=thickness, lineType=lineType)
    return image


def convert_torch_tensor_to_np(pred, gt, num_samples=None):
    def pred_to_np(p):      # Return a/two NxHxW numpy array
        p = F.softmax(p, dim=1)[:, 1, :, :]
        p = p.detach().cpu().numpy()
        binary_pred = np.array(p > 0.5, dtype='uint8')
        soft_pred = np.array(p)
        return binary_pred, soft_pred
    def gt_to_np(g):           # Return a NxHxW numpy array
        g = g.detach().cpu().numpy()
        g = np.array(g, dtype='uint8')
        return g
    bin_pred_np, soft_pred_np = pred_to_np(pred)
    gt_np = gt_to_np(gt)
    return bin_pred_np, soft_pred_np, gt_np

"""
Args:
    :dashboard: A tensorboardX summary write object.
    :group: The group to be written.
    :pred: Network output, Nx2xHxW shape tensor.
    :gt: Groundtruth.
Returns:
    None
"""
from torch.nn import functional as F
def export_netpred_gt_to_dashboard(dashboard, group, iteration, num_samples, pred, gt, visual=True):
    assert len(pred) == len(gt)
    assert len(pred) >= num_samples

    bin_pred_np, soft_pred_np, gt_np = convert_torch_tensor_to_np(pred, gt)
    
    if visual is True:
        for i, x in enumerate(zip(bin_pred_np, soft_pred_np, gt_np)):
            x = np.concatenate(x, axis=1)
            x = np.array(x * 255, dtype='uint8')
            x = np.repeat(x[..., None], 3, axis=2)
            dashboard.add_image("{}/{}".format(group, i), x, iteration)
            if i+1 >= num_samples: break

    return bin_pred_np, soft_pred_np, gt_np



    











