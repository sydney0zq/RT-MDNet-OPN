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

# Custom
from data_prov import VIDRegionDataset
from mdnet import MDNet, BinaryLoss, Precision
from opts.pretrain_opts import pretrain_opts
from logger.logger import setup_logger
sys.path.insert(0, "./maskrcnn-benchmark")
from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_fea_and_prop import get_opn
from maskrcnn_benchmark.modeling.poolers import Pooler


def set_optimizer(model, lr_base, lr_mult=pretrain_opts['lr_mult'], momentum=pretrain_opts['momentum'], w_decay=pretrain_opts['w_decay']):
    params = model.get_learnable_params()
    param_list = []
    for k, p in params.items():
        lr = lr_base
        for l, m in lr_mult.items():
            if k.startswith(l):
                lr = lr_base * m
        param_list.append({'params': [p], 'lr': lr})
    optimizer = torch.optim.SGD(param_list, lr=lr, momentum=momentum, weight_decay=w_decay)
    return optimizer

def sample_pos_neg_idxs(gt, rois, fg_thres=pretrain_opts['overlap_pos'][0], bg_thres=pretrain_opts['overlap_neg'][1],
                                    fg_num=pretrain_opts['batch_pos']*pretrain_opts['batch_frames'], 
                                    bg_num=pretrain_opts['batch_neg']*pretrain_opts['batch_frames']):
    if len(gt) != len(rois):
        assert False, "gt size {} is not same with rois size {}".format(len(gt), len(rois))
    gt = torch.from_numpy(gt).cuda()
    proposal_matcher = Matcher(fg_thres, bg_thres)

    total_matched_idxs = torch.LongTensor([]).cuda()
    for i_gt, i_roi in zip(gt, rois):
        i_gt = BoxList(i_gt[None, :], i_roi.size, mode="xywh")
        i_gt = i_gt.convert("xyxy")
        match_quality_matrix = boxlist_iou(i_gt, i_roi)
        matched_idxs = proposal_matcher(match_quality_matrix)
        total_matched_idxs = torch.cat([total_matched_idxs, matched_idxs])    # 0 is fg, -1 is bg, -2 is fg<>bg
    
    pos_idx = torch.nonzero(total_matched_idxs == 0).squeeze(1)
    neg_idx = torch.nonzero(total_matched_idxs == -1).squeeze(1)
    # randomly select positive and negative examples
    num_pos = min(pos_idx.numel(), fg_num)
    num_neg = min(neg_idx.numel(), bg_num)
    if len(pos_idx) >= fg_num:
        perm1 = torch.randperm(pos_idx.numel(), device=pos_idx.device)[:fg_num]
        perm2 = torch.randperm(neg_idx.numel(), device=neg_idx.device)[:bg_num]
    elif len(pos_idx) > 0:
        perm1 = torch.randint(0, pos_idx.size(0), (fg_num,)).type(torch.LongTensor)
        perm2 = torch.randint(0, neg_idx.size(0), (bg_num,)).type(torch.LongTensor)
    else:
        return None, None
    pos_idx = pos_idx[perm1]
    neg_idx = neg_idx[perm2]

    return pos_idx, neg_idx


def train_rtmdnet():
    # Prepare dataset, model, optimizer, loss
    with open(pretrain_opts['vid_pkl'], "rb") as fp:
        data = pickle.load(fp)
    K = len(data)
    K = 2000
    print ("VID has {} videos...".format(K))
    logger = setup_logger(logfile="./snapshots/train_{}cycle_{}video.log".format(pretrain_opts['n_cycles'], K))

    # Model
    mdnet = MDNet(K=K)
    mdnet.set_learnable_params(pretrain_opts['ft_layers'])
    mdnet.cuda()

    # https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/configs/e2e_mask_rcnn_R_50_FPN_1x.yaml
    # https://github.com/facebookresearch/maskrcnn-benchmark/blob/c56832ed8e05eb493c2e9ff8d8a8878a565223b9/maskrcnn_benchmark/modeling/poolers.py
    FPN_RoIAlign = Pooler(output_size=(7, 7),
                           scales=(0.25, 0.125, 0.0625, 0.03125),
                           sampling_ratio=2)
    # Optimizer
    binaryCriterion = BinaryLoss().cuda()
    evaluator = Precision()
    optimizer = set_optimizer(mdnet, pretrain_opts['lr'])
    
    # Data
    dataset = [None] * K
    print ("Building dataset...")
    for k, (seq_name, seq) in enumerate(data.items()):
        img_list, gt = seq['images'], seq['gt']
        img_dir = os.path.join(pretrain_opts['vid_home'], seq_name)
        dataset[k] = VIDRegionDataset(img_dir, img_list, gt, pretrain_opts)
        if k >= K-1: break

    best_score = 0
    batch_idx = 0
    precision = np.zeros(pretrain_opts['n_cycles'])
    for i in range(pretrain_opts['n_cycles']):
        print ("==== Start Cycle {} ====".format(i))
        k_list = np.random.permutation(K)
        prec = np.zeros(K)

        for j, k in enumerate(k_list):
            tic = time.time()
            img_pool, box_pool = dataset[k].__next__()
            #dataset[k].DEBUG_imgbox(img_pool, box_pool, "debug")
            opn_feat, opn_rois = mdnet.forward_OPN(img_pool)
            opn_roi_feats = FPN_RoIAlign(opn_feat, opn_rois)     #[Bx1000, 256, 7, 7]
            pos_idx, neg_idx = sample_pos_neg_idxs(box_pool, opn_rois)

            if pos_idx is None and neg_idx is None:
                continue

            pos_roi_feats = opn_roi_feats[pos_idx]
            neg_roi_feats = opn_roi_feats[neg_idx]
            
            pos_roi_feats = pos_roi_feats.view(pos_roi_feats.size(0), -1)
            neg_roi_feats = neg_roi_feats.view(neg_roi_feats.size(0), -1)

            # Compute score
            pos_score = mdnet(pos_roi_feats, k, in_layer='fc4')
            neg_score = mdnet(neg_roi_feats, k, in_layer='fc4')

            cls_loss = binaryCriterion(pos_score, neg_score)
            cls_loss.backward()
            batch_idx += 1

            if (batch_idx % pretrain_opts['seqbatch_size'])==0:
                print ("Update weights...")
                torch.nn.utils.clip_grad_norm_(mdnet.parameters(), pretrain_opts['grad_clip'])
                optimizer.step()
                optimizer.zero_grad()
                batch_idx = 0

            prec[k] = evaluator(pos_score, neg_score)

            toc = time.time() - tic
            if j % 10 == 0:
                print ("Cycle %2d, K %2d (%2d), BinLoss %.3f, Prec %.3f, Time %.3f" % \
                                              (i, j, k, cls_loss.item(), prec[k], toc))

        cur_score = prec.mean()
        precision[i] = cur_score
        print ("Mean Precision: %.3f " % (cur_score))
        if cur_score > best_score:
            best_score = cur_score
            states = {'fclayers': mdnet.fclayers.state_dict()}
            print ("Save model to %s" % pretrain_opts['model_path'])
            torch.save(states, pretrain_opts['model_path'])
    np.savetxt("precision.txt", precision, fmt='%2.2f')

if __name__ == "__main__":
    train_rtmdnet()

    

