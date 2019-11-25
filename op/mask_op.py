#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2019 qiang.zhou <theodoruszq@gmail.com>

"""

"""

import numpy as np


try:
    from mask_damaging import damage_masks
except:
    from .mask_damaging import damage_masks

"""
Args:
    :mask: A Nx1xHxW or HxW array.
Return:
    A Nx4 or 1x4 (minx, miny, maxx, maxy) box(es).
"""
def get_mask_bbox(m):
    if not np.any(m):
        return (0, 0, m.shape[1], m.shape[0])
    rows, cols = np.any(m, axis=1), np.any(m, axis=0)
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    h, w = m.shape
    ymin = max(0, ymin)
    ymax = min(h-1, ymax)
    xmin = max(0, xmin)
    xmax = min(w-1, xmax)
    if xmin == xmax or ymin == ymax:
        return (0, 0, m.shape[1], m.shape[0])
    else:
        return (xmin, ymin, xmax, ymax)

def get_B_mask_bbox(mask):
    assert mask.ndim == 4
    return [ get_mask_bbox(np.squeeze(m)) for m in mask]

"""
Args:
    :mask: A mask HxW array.
    :box: A (minx, miny, maxx, maxy) box.
Returns:
    A cropped mask/image.
"""
def get_cropped_mask(image, box):
    image = np.array(image)
    minx, miny, maxx, maxy = [int(x) for x in box]
    width, height = max(0, maxx - minx), max(0, maxy - miny)
    assert (not(width == 0 or height == 0)), \
            "Nonsense cropbox: ({}, {}, {}, {})...".format(minx, miny, maxx, maxy)

    roi_left = max(0, min(minx, image.shape[1]))
    roi_right = min(image.shape[1], max(0, maxx))
    roi_top = max(0, min(miny, image.shape[0]))
    roi_bottom = min(image.shape[0], max(0, maxy))
    roi_width, roi_height = roi_right-roi_left, roi_bottom-roi_top
    pad_x, pad_y = max(0, 0-minx), max(0, 0-miny)
    
    #print (roi_left, roi_top, roi_right, roi_bottom)
    #print (roi_height, roi_width)
    if len(image.shape) == 3:
        padimage = np.zeros((height, width, 3))
        padimage[pad_y:pad_y+roi_height, pad_x:pad_x+roi_width, :] = image[roi_top:roi_bottom, roi_left:roi_right, :]
    elif len(image.shape) == 2:
        padimage = np.zeros((height, width))
        padimage[pad_y:pad_y+roi_height, pad_x:pad_x+roi_width] = image[roi_top:roi_bottom, roi_left:roi_right]

    #import cv2
    #cv2.imwrite('padimage.jpg', padimage)
    #print (padimage.shape)
    return padimage

"""
Args:
    :masks: A Nx1xHxW np array.
    :boxes: A list with N boxes.
Return:
    A list with [1x1xH'xW', 1x1xH''xW'' ...]
"""
def get_B_cropped_mask(masks, boxes):
    assert masks.ndim == 4 and len(masks) == len(boxes) and masks.shape[1] == 1
    cropped_masks= [ get_cropped_mask(np.squeeze(mask), box)[None, None, ...] 
                                        for mask, box in zip(masks, boxes) ]
    
    return cropped_masks


get_damage_mask = damage_masks

"""
Args:
    :masks: Must be a [1x1xHxW, 1x1xH'xW' ...] list.
"""
def get_B_damage_mask(masks):
    damage_mask = []
    for mask in masks:
        if np.sum(mask) != 0:
            damage_mask.append(get_damage_mask(np.squeeze(mask)))
        else:
            damage_mask.append(mask)
    return damage_mask
    #return [ get_damage_mask(np.squeeze(mask)) for mask in masks ]


from torch.nn import functional as F
import torch

"""
Args:
    :feat: A list with [Nx256xRHxRW, Nx256xRH'xRW', Nx256xRH''xRW'' ...]
    :mask: A list with [1x1xMHxMW, 1x1xMH'xMW' ...] -> N
Return:
    A list with feat and mask concated along the channel axis.
"""
def dist_scale_and_concat_mask(feat, mask):
    assert feat[0].size(0) == len(mask)
    dtype, device = feat[0].dtype, feat[0].device

    sizes = [(i_feat.size(2), i_feat.size(3)) for i_feat in feat]

    mask_ = []
    for hw in sizes:
        size_mask = []
        for i_mask in mask:
            i_mask = torch.from_numpy(i_mask).type(dtype).to(device)
            i_mask = F.interpolate(i_mask, size=hw, mode="nearest")
            size_mask.append(i_mask)
        mask_.append(torch.cat(size_mask, dim=0))

    feat = [ torch.cat([feat[i], i_mask], dim=1)
                for i, i_mask in enumerate(mask_) ]
    return feat


import scipy.ndimage as nd
def get_resized_mask(mask, in_size, out_size):
    if len(mask.shape) == 3:
        mask_ = nd.zoom(mask.astype('uint8'),
                (1.0, out_size[0]/float(in_size[0]), out_size[1]/float(in_size[1])))
    elif len(mask.shape) == 4:
        mask_ = nd.zoom(mask.astype('uint8'),
                (1.0, 1.0, out_size[0]/float(in_size[0]), out_size[1]/float(in_size[1])))
    else:
        mask_ = nd.zoom(mask.astype('uint8'),
                (out_size[0]/float(in_size[0]), out_size[1]/float(in_size[1])))
    return mask_


from torch.nn import functional as F
import torch

def get_B_resized_torch_mask(mask, size):
    assert len(mask[0].shape) == 4
    torch_mask = []
    for i_mask in mask:
        i_mask = F.interpolate(torch.from_numpy(i_mask),
                               size=size, 
                               mode="nearest").squeeze(dim=1).type('torch.cuda.LongTensor')
        torch_mask.append(i_mask)
    return torch.cat(torch_mask, dim=0)

# Calc IOU
def get_mask_iou(pred, gt, obj_n):
    assert(gt.shape == pred.shape)
    ious = np.zeros((obj_n), dtype=np.float32)
    for obj_id in range(1, obj_n+1):
        gt_mask = gt == obj_id
        pred_mask = pred == obj_id
        inter = gt_mask & pred_mask
        union = gt_mask | pred_mask

        if union.sum() == 0:
            ious[obj_id-1] = 1
        else:
            ious[obj_id-1] = float(inter.sum()) / union.sum()
    return ious

"""
Args:
    :pred: A NxHxW np array
    :gt: A NxHxW np array.
"""
def get_B_mask_iou(pred, gt, obj_n=1):
    assert len(pred) == len(gt)
    return [ get_mask_iou(i_pred, i_gt, obj_n)
                for i_pred, i_gt in zip(pred, gt) ]
    
if __name__ == "__main__":
    a = np.ones((5, 10, 10))
    b = np.zeros((5, 10, 10))
    print (get_B_mask_iou(a, b, 1))
    exit()


        




if __name__ == "__main__":
    #mask = np.zeros((5, 1, 10, 10))
    #mask[0, 0, 1:4, 2:8] = 1
    #print (get_mask_bbox(mask))
    #r = [torch.ones((1, 256, 60, 70)), torch.ones((1, 256, 40, 50))]
    #mask = np.zeros((1, 1, 100, 100))
    #ret = dist_scale_and_concat_mask(r, mask)
    feat = [torch.zeros((2, 256, 60, 60)), torch.zeros((2, 256, 30, 30))]
    masks = [np.ones((1, 100, 100)), np.ones((1, 80, 80))]
    #boxes = [(1, 1, 10, 10), (2, 2, 80, 80)]
    #get_B_cropped_mask(masks, boxes)
    #get_B_damage_masks(masks)
    tt = dist_scale_and_concat_mask(feat, masks)
    import pdb
    pdb.set_trace()




