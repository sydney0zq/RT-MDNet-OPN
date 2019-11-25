#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2019 qiang.zhou <theodoruszq@gmail.com>

"""

"""

import json
import numpy as np

def scalebox(box, scale):
    if box.ndim == 1: box = box[None, :]
    minx, miny, maxx, maxy = box[:, 0], box[:, 1], box[:, 2], box[:, 3]
    w, h = maxx-minx, maxy-miny
    cx, cy = (maxx+minx)/2., (maxy+miny)/2.
    new_w, new_h = w*scale, h*scale
    new_minx, new_miny = cx-new_w/2., cy-new_h/2.
    new_maxx, new_maxy = cx+new_w/2., cy+new_h/2.
    return np.stack([new_minx, new_miny, new_maxx, new_maxy], axis=1)


from data_prov import DAVISRegionDataset

davis = DAVISRegionDataset({"img_size": (864, 480), "mean_value": [0, 0, 0]})

with open("result.json", "r") as f:
    result = json.load(f)

iou_array = []

# Input box should in (minx, miny, maxx, maxy)
def overlap_ratio_crossbox(rect1, rect2):
    if rect1.ndim == 1: rect1 = rect1[None, :]
    if rect2.ndim == 1: rect2 = rect2[None, :]

    left = np.maximum(rect1[:,0], rect2[:,0])
    right = np.minimum(rect1[:,2], rect2[:,2])
    top = np.maximum(rect1[:,1], rect2[:,1])
    bottom = np.minimum(rect1[:,3], rect2[:,3])

    intersect = np.maximum(0, right - left) * np.maximum(0, bottom - top)
    union = (rect1[:,2]-rect1[:,0])*(rect1[:,3]-rect1[:,1]) + (rect2[:,2]-rect2[:,0])*(rect2[:,3]-rect2[:,1]) - intersect
    iou = np.clip(intersect / union, 0, 1)
    return iou


# Mean IOU
# RT-MDNet + OPN shared
#for imgs, boxes, seq_name, label_id in davis:
#    key = "{}-{}".format(seq_name, label_id)
#    predict = result[key]
#    assert len(boxes) == len(predict)
#    iou = overlap_ratio_crossbox(np.array(predict), boxes)
#    print (iou)
#    iou_array.append(np.mean(iou))

# rt-mdnet + opn cascade
#for imgs, boxes, seq_name, label_id in davis:
#    key = "{}-{}".format(seq_name, label_id)
#    try:
#        with open("result_davis/{}/{}/result.json".format(seq_name, label_id)) as f:
#            predict = np.array(json.load(f)["res"])
#            predict[:, 2:] += predict[:, :2]
#        import pdb
#        pdb.set_trace()
#        assert len(boxes) == len(predict)
#        iou = overlap_ratio_crossbox(np.array(predict), boxes)
#    except:
#        iou = np.array([0])
#    print (iou)
#    iou_array.append(np.mean(iou))
#
#print (np.mean(np.array(iou_array)))

def include_check(gt, predict):
    inc_minx = gt[:, 0] >= predict[:, 0]
    inc_miny = gt[:, 1] >= predict[:, 1]
    inc_maxx = gt[:, 2] <= predict[:, 2]
    inc_maxy = gt[:, 3] <= predict[:, 3]

    inc_array = inc_minx*inc_miny*inc_maxx*inc_maxy
    iou_array = overlap_ratio_crossbox(gt, predict) > 0.5

    or_array = inc_array | iou_array
    return np.sum(inc_array)



# Mean Recall
# RT-MDNet + OPN shared
#total_recall = []
#for imgs, boxes, seq_name, label_id in davis:
#    key = "{}-{}".format(seq_name, label_id)
#    predict = np.array(result[key])
#    assert len(boxes) == len(predict)
#    expand_pred = scalebox(predict, 1.5)
#    #iou = overlap_ratio_crossbox(predict, boxes)
#    recall = include_check(boxes, expand_pred)
#    print (recall*1.0/len(imgs))
#    total_recall.append(recall*1.0/len(imgs))
#
#    #iou_array.append(np.mean(iou))

# rt-mdnet + opn cascade
total_recall = []
for imgs, boxes, seq_name, label_id in davis:
    key = "{}-{}".format(seq_name, label_id)
    try:
        with open("result_davis/{}/{}/result.json".format(seq_name, label_id)) as f:
            predict = np.array(json.load(f)["res"])
            predict[:, 2:] += predict[:, :2]
        expand_pred = scalebox(np.array(predict), 1.5)
        assert len(boxes) == len(predict)
        #iou = overlap_ratio_crossbox(np.array(predict), boxes)
        recall = include_check(boxes, expand_pred)
        recall = recall * 1.0 / len(imgs)
    except:
        recall = 0
    print (recall)
    total_recall.append(recall)

#print (np.mean(np.array(iou_array)))
print ("-------------------------------")
print (np.mean(np.array(total_recall)))


