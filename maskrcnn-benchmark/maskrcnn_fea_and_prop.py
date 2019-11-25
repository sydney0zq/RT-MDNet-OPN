#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2019 qiang.zhou <theodoruszq@gmail.com>

"""
Output maskrcnn backbone feature and RPN results.
"""

import torch
import torch.nn as nn
from torchvision import transforms as T
import cv2

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer

import os
CURR_DIR = os.path.dirname(os.path.realpath(__file__)) 

# Generalized Object proposal network
class OPN(nn.Module):
    def __init__(self, cfg, image_size=(100, 100)):
        super(OPN, self).__init__()
        self.cfg = cfg.clone()
        self.model = build_detection_model(cfg)
        self.model.eval()
        self.backbone = self.model.backbone
        self.rpn = self.model.rpn
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.model.to(self.device)
        self.image_size = image_size        # (h, w)
    
        checkpointer = DetectronCheckpointer(cfg, self.model)
        _ = checkpointer.load(cfg.MODEL.WEIGHT)

        self.transforms = self.build_transform()

        self.cpu_device = torch.device("cpu")

    def build_transform(self):
        """
        Creates a basic transformation that was used to train the models.
        """
        cfg = self.cfg
        # We need BGR format image.
        if cfg.INPUT.TO_BGR255:
            to_bgr_transform = T.Lambda(lambda x: x * 255)
        else:
            to_bgr_transform = T.Lambda(lambda x: x[[2, 1, 0]])

        normalize_transform = T.Normalize(
                mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)

        transform = T.Compose(
                    [
                        T.ToPILImage(),
                        T.Resize(self.image_size),
                        T.ToTensor(),
                        to_bgr_transform,
                        normalize_transform,
                    ]
                )
        return transform

    def np2tensor(self, images):
        images = torch.from_numpy(images).float()
        return images

    # images is in Nx3xHxW BGR format and they don't need to resize or subtract too mean value anymore
    def forward(self, images):
        #images = self.transforms(images)
        images = self.np2tensor(images)
        #image_list = to_image_list(images, self.cfg.DATALOADER.SIZE_DIVISIBILITY)
        image_list = to_image_list(images)
        image_list = image_list.to(self.device)
        with torch.no_grad():
            features, proposals = self.model.forward_rpn(image_list)
        return features, proposals
    

def get_opn(prefix=CURR_DIR):
    config_file = "{}/configs/e2e_mask_rcnn_R_50_FPN_1x.yaml".format(prefix)
    cfg.merge_from_file(config_file)
    cfg.merge_from_list(['MODEL.WEIGHT', "{}/e2e_mask_rcnn_R_50_FPN_1x.pth".format(prefix)])
    opn = OPN(cfg)
    return opn


if __name__ == "__main__":
    config_file = "./configs/e2e_mask_rcnn_R_50_FPN_1x.yaml"
    cfg.merge_from_file(config_file)
    cfg.merge_from_list(['MODEL.WEIGHT', "./e2e_mask_rcnn_R_50_FPN_1x.pth"])

    opn = OPN(cfg)
    aa = cv2.imread('/tmp/00000.jpg')
    import numpy as np
    aa = np.array((10, 3, 224, 224))
    opn(aa)
    import pdb
    pdb.set_trace()
    #import numpy as np
    #aa = np.expand_dims(aa, axis=0)
    #aa = torch.ones((1, 3, 480, 910))
    #aa = torch.ones((1, 3, 107, 107))
    #aa = aa.cuda()
    ff, prop = opn(aa)
    #print (prop)
    import sys
    print(sys.path)
    import pdb
    pdb.set_trace()







