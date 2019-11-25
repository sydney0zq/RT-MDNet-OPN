#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2019 qiang.zhou <theodoruszq@gmail.com>

# Public
import pickle
import os
import numpy as np
import sys
import torch
import time
from tensorboardX import SummaryWriter

# Custom
from data_prov import YVOSRegionDataset
from drsn import DRSN, CrossEntropyLoss_Softmax
from op.select_op import select_image_mask_pair, get_sample_ids
from op.mask_op import get_B_mask_bbox, get_cropped_mask, get_damage_mask, get_resized_mask
from op.mask_op import get_B_cropped_mask, get_B_damage_mask, get_B_resized_torch_mask, dist_scale_and_concat_mask, get_B_mask_iou
from op.box_op import get_expand_box, get_shaked_box, get_B_expand_box, get_B_shaked_box
from op.visual_op import export_netpred_gt_to_dashboard, convert_torch_tensor_to_np
from op.system_op import get_function_file

from opts.pretrain_opts import pretrain_opts
from logger.logger import setup_logger
sys.path.insert(0, "./maskrcnn-benchmark")
from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_fea_and_prop import get_opn
from maskrcnn_benchmark.modeling.poolers import Pooler, Encoder_Pooler


logger = get_function_file(setup_logger, pretrain_opts['log_file'])

opnfeat_pooler = Encoder_Pooler(output_sizes=[63, 32, 16, 8], 
                                scales=(0.25, 0.125, 0.0625, 0.03125),
                                sampling_ratio=2)

fapply = lambda f, x: [f(ix) for ix in x]

def get_encoder_output(sample, opn):
    imgs, masks, seq_name, label_id = sample
    ref_imgs, ref_masks, cur_imgs, cur_masks = select_image_mask_pair(imgs, masks, pair_num=8)
    scene_size = ref_imgs.shape[-2:]
    ref_opn_feat, _ = opn(ref_imgs)
    cur_opn_feat, _ = opn(cur_imgs)
    ref_gtbox, cur_gtbox = get_B_mask_bbox(ref_masks), get_B_mask_bbox(cur_masks)

    # Pre-Processing to get cropped Encoder feature
    ref_enlargebox = get_B_expand_box(ref_gtbox, scene_size, ratio=1.5)
    cur_simbox = get_B_expand_box(get_B_shaked_box(cur_gtbox, scene_size), scene_size, ratio=1.5)
    ref_encfeat = opnfeat_pooler(ref_opn_feat, torch.from_numpy(np.array(ref_enlargebox)))
    cur_encfeat = opnfeat_pooler(cur_opn_feat, torch.from_numpy(np.array(cur_simbox)))
    
    # Pad and concat masks
    ref_enlargemask = get_B_cropped_mask(ref_masks, ref_enlargebox)
    cur_simmask = get_B_cropped_mask(cur_masks, cur_simbox)
    prev_simmask = get_B_damage_mask(cur_simmask)
    ref_encfeat = dist_scale_and_concat_mask(ref_encfeat, ref_enlargemask)
    cur_encfeat = dist_scale_and_concat_mask(cur_encfeat, prev_simmask)

    # Make groundtruth
    cur_gtmask = get_B_resized_torch_mask(cur_simmask, (256, 256))
    
    return {"feat": [ref_encfeat, cur_encfeat],
            "gt": cur_gtmask}

def valid_drsn(val_set, val_ids, opn, drsn):
    valid_iou = []
    for val_iter, idx in enumerate(val_ids):
        with torch.no_grad():
            ret_dict = get_encoder_output(val_set[idx], opn)
            ref_encfeat, cur_encfeat = ret_dict['feat']
            cur_gtmask = ret_dict['gt']
            # Feed into decoder and optimize
            cur_pred_mask = drsn(ref_encfeat, cur_encfeat)
            bin_pred_np, soft_pred_np, gt_np = convert_torch_tensor_to_np(cur_pred_mask, cur_gtmask)
            iou = np.mean(np.array(get_B_mask_iou(bin_pred_np, gt_np)))
            logger.info ("Val Iteration {}, DRSN_IoU {}".format(val_iter, iou))
            valid_iou.append(iou)
    return np.mean(np.array(valid_iou))

def train_drsn():
    train_set = YVOSRegionDataset(pretrain_opts, "train")
    val_set   = YVOSRegionDataset(pretrain_opts, "val")
    train_ids = get_sample_ids(pretrain_opts['drsn_n_cycles'], len(train_set), rand=True)
    #train_ids = get_sample_ids(1, len(train_set), rand=False)
    print (len(train_set))
    val_ids   = get_sample_ids(1, len(val_set), rand=False)

    opn = get_opn()
    opn.eval()
    opn = opn.cuda()

    drsn = DRSN()
    drsn = drsn.cuda()
    drsn.set_bn_status(freeze=False)
    logger.info (drsn)

    criterion = CrossEntropyLoss_Softmax(reduction='mean').cuda()

    optimizer = torch.optim.Adam(drsn.parameters(), lr=1e-3)
    optimizer.zero_grad()

    dashboard = SummaryWriter(log_dir="dashboard")

    # First train with only one batch, note add validate process
    for iteration, idx in enumerate(train_ids):
        ret_dict = get_encoder_output(train_set[idx], opn)
        ref_encfeat, cur_encfeat = ret_dict['feat']
        cur_gtmask = ret_dict['gt']
        
        # Feed into decoder and optimize
        cur_pred_mask = drsn(ref_encfeat, cur_encfeat)

        loss = criterion(cur_pred_mask, cur_gtmask)
        loss.backward()
        logger.info ("Iteration {}, DRSN CELoss {}".format(iteration, loss.item()))

        # Visualize masks and IoU
        if iteration % pretrain_opts["verbose_freq"] == 0:
            dashboard.add_scalar("DRSN_Loss/train", loss.item(), iteration)
            bin_pred_np, soft_pred_np, gt_np = export_netpred_gt_to_dashboard(dashboard, "DRSN_Mask", iteration, 
                                                   pretrain_opts['batch_frames']//2, 
                                                   cur_pred_mask, cur_gtmask)
            train_iou = np.mean(np.array(get_B_mask_iou(bin_pred_np, gt_np)))
            dashboard.add_scalar("DRSN_IoU/train", train_iou, iteration)

        # Dump snapshots
        if iteration % pretrain_opts['dump_freq'] == 0:
            os.makedirs(pretrain_opts['drsn_snapshot_home'], exist_ok=True)
            model_fn = os.path.join(pretrain_opts['drsn_snapshot_home'], "drsn_{:06d}.pth".format(iteration))
            torch.save(drsn.state_dict(), model_fn)
            logger.info ("DRSN model {} saved...".format(model_fn))
        
        # Visualize validation IoU
        if iteration % pretrain_opts['valid_freq'] == 0 and iteration > 0:
            valid_iou = valid_drsn(val_set, val_ids, opn, drsn)
            dashboard.add_scalar("DRSN_IoU/val", valid_iou, iteration)

        # Optimize
        if iteration % pretrain_opts['drsn_seqbatch_size'] == 0 and iteration > 0:
            torch.nn.utils.clip_grad_norm_(drsn.parameters(), pretrain_opts['grad_clip'])
            optimizer.step()
            optimizer.zero_grad()


if __name__ == "__main__":
    train_drsn()

    

