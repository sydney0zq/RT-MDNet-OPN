#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2019 qiang.zhou <theodoruszq@gmail.com>

"""

"""
from maskrcnn_benchmark.config import cfg

config_file = "./configs/e2e_mask_rcnn_R_50_FPN_1x.yaml"

cfg.merge_from_file(config_file)
cfg.merge_from_list(['MODEL.WEIGHT', "./e2e_mask_rcnn_R_50_FPN_1x.pth"])
print (cfg)

from demo.predictor import COCODemo

coco_demo = COCODemo(cfg, min_image_size=800, confidence_threshold=0.7)
import cv2
image = cv2.imread("/tmp/00000.jpg")
predictions = coco_demo.run_on_opencv_image(image)


import pdb
pdb.set_trace()

