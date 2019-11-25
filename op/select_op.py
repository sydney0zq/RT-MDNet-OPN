#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2019 qiang.zhou <theodoruszq@gmail.com>

"""

"""

import numpy as np

"""
Args:
    :imgs: A Nx3xHxW image array.
    :masks: A Nx1xHxW mask array.
    :pair_num: The number of image-mask pair.

Return:
    Selected ref imgs, ref_masks and cur_imgs, cur_masks.
"""
def select_image_mask_pair(imgs, masks, pair_num):
    assert len(imgs) == len(masks)
    pair_num = int(pair_num)
    empty_mask_ids = select_empty_mask_index(masks)
    ref_domain = [x for x in range(len(imgs)) if x not in empty_mask_ids]   # Remove empty masks index

    ref_ids = np.random.choice(ref_domain, size=pair_num, replace=pair_num>len(ref_domain))
    cur_ids = np.random.choice(len(imgs), size=pair_num, replace=pair_num>len(imgs))

    #print (ref_ids, cur_ids)
    ref_imgs, ref_masks = imgs[ref_ids], masks[ref_ids]
    cur_imgs, cur_masks = imgs[cur_ids], masks[cur_ids]
    return [ref_imgs, ref_masks, cur_imgs, cur_masks]

"""
Args:
    :masks: A Nx1xHxW mask array.
Return:
    An index vector to show which mask are empty/less than 50 pixels.
"""
def select_empty_mask_index(masks, minimum_cnt=50):
    pixel_cnt = np.sum(masks, axis=(1, 2, 3))
    empty_bool = pixel_cnt < minimum_cnt
    empty_index = np.nonzero(empty_bool)[0]
    return empty_index

import random

def get_sample_ids(cycles, length, rand=True):
    sample_ids = []
    for i in range(cycles):
        ids = [*(range(length))]
        if rand is True:
            random.shuffle(ids)
        sample_ids += ids
    return sample_ids



if __name__ == "__main__":
    a = np.ones((4, 3, 100, 100))
    aa = np.ones((4, 1, 100, 100))
    aa[1:, :, :, :] = 0
    #select_image_mask_pair(a, aa, 2)
    aa = get_sample_ids(10, 20, True)

    
    #ra, rb, ca, cb = select_image_mask_pair(a, b, pair_num=5)
    #print (ra.shape, rb.shape)
    #print (ca.shape, cb.shape)
    import pdb
    pdb.set_trace()



