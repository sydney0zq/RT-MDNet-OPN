#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2019 qiang.zhou <theodoruszq@gmail.com>

"""
Utils.
"""

import numpy as np

def scalebox(box, scale):
    minx, miny, maxx, maxy = box[0], box[1], box[2], box[3]
    w, h = max(maxx-minx, 1), max(maxy-miny, 1)
    cx, cy = (maxx+minx)/2., (maxy+miny)/2.
    new_w, new_h = w*scale, h*scale
    new_minx, new_miny = cx-new_w/2., cy-new_h/2.
    new_maxx, new_maxy = cx+new_w/2., cy+new_h/2.
    return np.array([new_minx, new_miny, new_maxx, new_maxy])
