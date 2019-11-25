#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2019 qiang.zhou <theodoruszq@gmail.com>


import numpy as np


"""
Args:
    :box: A 4, array.
    :scene_size: The shape of scene, in (h, w) format.
    :floor: The coordinates are integer.
Return:
    Expanded box.
"""
def get_expand_box(box, scene_size, ratio=1.5, floor=True):
    minx, miny, maxx, maxy = box
    cx, cy = (minx+maxx)/2., (miny+maxy)/2.
    h, w = maxy-miny, maxx-minx
    minx_ = max(0, cx-ratio*w/2.0)
    miny_ = max(0, cy-ratio*h/2.0)
    maxx_ = min(scene_size[1], max(minx_, cx+ratio*w/2.0))
    maxy_ = min(scene_size[0], max(miny_, cy+ratio*h/2.0))
    if floor is True:
        return (int(minx_), int(miny_), int(maxx_), int(maxy_))
    else:
        return (minx_, miny_, maxx_, maxy_)

def get_B_expand_box(boxes, scene_size, ratio=1.5, floor=True):
    return [get_expand_box(box, scene_size, ratio=ratio, floor=floor) for box in boxes]



def get_shaked_box(box, scene_size, trans_f=0.3, scale_f=1.1, floor=True):
    minx, miny, maxx, maxy = box
    cx, cy = (minx+maxx)/2., (miny+maxy)/2.
    h, w = maxy-miny, maxx-minx
    #samples[:,:2] += trans_f * np.mean(bb[2:]) * np.clip(0.5*np.random.randn(n,2),-1,1)
    #samples[:,2:] *= scale_f ** np.clip(0.5*np.random.randn(n,1),-1,1)
    cx += trans_f * (h+w)/2. * np.clip(0.5*np.random.randn(), -1, 1)
    cy += trans_f * (h+w)/2. * np.clip(0.5*np.random.randn(), -1, 1)
    h *= scale_f ** np.clip(0.5*np.random.randn(), -1, 1)
    w *= scale_f ** np.clip(0.5*np.random.randn(), -1, 1)
    minx_ = max(0, cx-w/2.0)
    miny_ = max(0, cy-h/2.0)
    maxx_ = min(scene_size[1], max(minx_, cx+w/2.0))
    maxy_ = min(scene_size[0], max(miny_, cy+h/2.0))

    if minx_ >= maxx_ or miny_ >= maxy_:
        minx_, miny_ = 0, 0
        maxx_, maxy_ = scene_size[1], scene_size[0]
    if floor is True:
        return (int(minx_), int(miny_), int(maxx_), int(maxy_))
    else:
        return (minx_, miny_, maxx_, maxy_)

def get_B_shaked_box(boxes, scene_size, trans_f=0.3, scale_f=1.1, floor=True):
    return [ get_shaked_box(box, scene_size, trans_f=trans_f, scale_f=scale_f, floor=floor) 
                          for box in boxes ]


if __name__ == "__main__":
    #from visual_op import overlay_box
    #import cv2
    #image = np.zeros((200, 200, 3))
    #box = np.array([80, 80, 120, 180])
    #image_ = overlay_box(image, box, color=(0, 0, 255))
    #for i in range(10):
    #    s_box = get_shaked_box(box, (200, 200), 0.3, 1.1)
    #    s_box = get_expand_box(s_box, (200, 200), ratio=1.5)
    #    image_ = overlay_box(image_, np.array(s_box))
    #cv2.imwrite("shaked_box.jpg", image_)
    box = np.array([1, 1, 3, 3])
    scene_size = (100, 100)
    i = 0
    while True:
        if i % 100 == 0:
            print (i)
        i+= 1
        minx, miny, maxx, maxy = get_shaked_box(box, scene_size)
        if minx >= maxx or miny >=  maxy:
            print ("error")
        if i > 100000:
            break



