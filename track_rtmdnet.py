#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2019 qiang.zhou <theodoruszq@gmail.com>

import torch
import numpy as np
import os
import cv2
import json
import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from data_prov import DAVISRegionDataset
from sample_generator import gen_samples, SampleGenerator
from mdnet import MDNet, BinaryLoss
from opts.track_opts import opts
from misc import scalebox
import sys
sys.path.insert(0, "./maskrcnn-benchmark")
from maskrcnn_benchmark.modeling.poolers import Pooler
from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou

depreprocess = lambda img: cv2.cvtColor(np.uint8((img + np.array(opts['mean_value']).reshape((3, 1, 1))).transpose((1, 2, 0))), cv2.COLOR_BGR2RGB)
boxarea = lambda box: (box[2]-box[0])*(box[3]-box[1])

def set_optimizer(model, lr_base, lr_mult=opts['lr_mult'], momentum=opts['momentum'], w_decay=opts['w_decay']):
    params = model.get_learnable_params()
    param_list = []
    for k, p in params.items():
        lr = lr_base
        for l, m in lr_mult.items():
            if k.startswith(l):
                lr = lr_base * m
        param_list.append({'params': [p], 'lr':lr})
    optimizer = torch.optim.SGD(param_list, lr = lr, momentum=momentum, weight_decay=w_decay)
    return optimizer

def sample_pos_neg_idxs(gt, rois, fg_thres=0.7, bg_thres=0.3, fg_num=32, bg_num=96):
    if len(gt) != len(rois):
        assert False, "gt size {} is not same with rois size {}".format(len(gt), len(rois))
    gt = torch.from_numpy(gt).cuda()
    proposal_matcher = Matcher(fg_thres, bg_thres)

    total_matched_idxs = torch.LongTensor([]).cuda()
    for i_gt, i_roi in zip(gt, rois):
        i_gt = BoxList(i_gt[None, :], i_roi.size, mode="xyxy")
        #i_gt = i_gt.convert("xyxy")
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

# prevbox and rois are torch.Tensor with Nx4 size
def sample_cand_idxs(prevbox, rois, size, thres=0.3):
    prevbox = torch.from_numpy(prevbox).cuda()
    proposal_matcher = Matcher(thres, 0)    # fg thres; bg thres

    total_matched_idxs = torch.LongTensor([]).cuda()
    prevbox = BoxList(prevbox, size, mode="xyxy")
    rois = BoxList(rois, size, mode="xyxy")

    match_quality_matrix = boxlist_iou(prevbox, rois)
    matched_idxs = proposal_matcher(match_quality_matrix)
    
    pos_idx = torch.nonzero(matched_idxs == 0).squeeze(1)

    if len(pos_idx) >= 5:
        return pos_idx
    elif len(pos_idx) > 0:
        perm = torch.randint(0, pos_idx.size(0), (10,)).type(torch.LongTensor)
        return pos_idx[perm]
    else:
        return None

def sample_pos_neg_idxs_4tensor(prevbox, rois, size, fg_thres=0.7, bg_thres=0.3, fg_num=32, bg_num=96):
    prevbox = torch.from_numpy(prevbox).cuda()
    proposal_matcher = Matcher(fg_thres, bg_thres)    # fg thres; bg thres

    total_matched_idxs = torch.LongTensor([]).cuda()
    prevbox = BoxList(prevbox, size, mode="xyxy")
    rois = BoxList(rois, size, mode="xyxy")

    match_quality_matrix = boxlist_iou(prevbox, rois)
    matched_idxs = proposal_matcher(match_quality_matrix)
    
    pos_idx = torch.nonzero(matched_idxs == 0).squeeze(1)
    neg_idx = torch.nonzero(matched_idxs == -1).squeeze(1)

    # randomly select positive and negative examples
    num_pos = min(pos_idx.numel(), fg_num)
    num_neg = min(neg_idx.numel(), bg_num)
    if len(pos_idx) >= fg_num and len(neg_idx) >= bg_num:
        perm1 = torch.randperm(pos_idx.numel(), device=pos_idx.device)[:fg_num]
        perm2 = torch.randperm(neg_idx.numel(), device=neg_idx.device)[:bg_num]
    elif len(pos_idx) > 0 and len(neg_idx) > 0:
        perm1 = torch.randint(0, pos_idx.size(0), (fg_num,)).type(torch.LongTensor)
        perm2 = torch.randint(0, neg_idx.size(0), (bg_num,)).type(torch.LongTensor)
    else:
        return None, None

    pos_idx = pos_idx[perm1]
    neg_idx = neg_idx[perm2]

    return pos_idx, neg_idx



def train(model, criterion, optimizer, pos_feats, neg_feats, maxiter, in_layer='fc4'):
    model.fclayers.train()

    batch_pos = opts['batch_pos']
    batch_neg = opts['batch_neg']
    batch_test = opts['batch_test']
    batch_neg_cand = max(opts['batch_neg_cand'], batch_neg)

    pos_idx = np.random.permutation(pos_feats.size(0))
    neg_idx = np.random.permutation(neg_feats.size(0))
    while(len(pos_idx) < batch_pos*maxiter):
        pos_idx = np.concatenate([pos_idx, np.random.permutation(pos_feats.size(0))])
    while(len(neg_idx) < batch_neg_cand*maxiter):
        neg_idx = np.concatenate([neg_idx, np.random.permutation(neg_feats.size(0))])
    pos_pointer = 0
    neg_pointer = 0

    for iter in range(maxiter):

        # select pos idx
        pos_next = pos_pointer+batch_pos
        pos_cur_idx = pos_idx[pos_pointer:pos_next]
        pos_cur_idx = pos_feats.new(pos_cur_idx).long()
        pos_pointer = pos_next

        # select neg idx
        neg_next = neg_pointer+batch_neg_cand
        neg_cur_idx = neg_idx[neg_pointer:neg_next]
        neg_cur_idx = neg_feats.new(neg_cur_idx).long()
        neg_pointer = neg_next

        # create batch
        batch_pos_feats = pos_feats.index_select(0, pos_cur_idx)
        batch_neg_feats = neg_feats.index_select(0, neg_cur_idx)

        # hard negative mining
        if batch_neg_cand > batch_neg:
            model.fclayers.eval() ## model transfer into evaluation mode
            for start in range(0,batch_neg_cand,batch_test):
                end = min(start+batch_test,batch_neg_cand)
                score = model(batch_neg_feats[start:end], in_layer=in_layer)
                if start==0:
                    neg_cand_score = score.data[:,1].clone()
                else:
                    neg_cand_score = torch.cat((neg_cand_score, score.data[:,1].clone()),0)

            _, top_idx = neg_cand_score.topk(batch_neg)
            batch_neg_feats = batch_neg_feats.index_select(0, top_idx)
            model.fclayers.train() ## model transfer into train mode

        # forward
        pos_score = model(batch_pos_feats, in_layer=in_layer)
        neg_score = model(batch_neg_feats, in_layer=in_layer)

        # optimize
        loss = criterion(pos_score, neg_score)
        model.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), opts['grad_clip'])
        optimizer.step()
        
        #if iter % 10 == 0:
        #    print ("Iter %d, Loss %.4f" % (iter, loss.item()))

# image is in HxWxC shape
# box is in Nx4 shape. (minx, miny, maxx, maxy)
# gt is in 1x4 shape, (minx, miny, maxx, maxy)
def overlay_box(image, box, gt=None, boxcolor="ff0000", savefig_path=None):
    assert image.shape[2] == 3
    if len(box.shape) == 1 and len(box) == 4: box = box[None, :]
    dpi = 80.0
    figsize = (image.shape[1]/dpi, image.shape[0]/dpi)

    fig = plt.figure(frameon=False, figsize=figsize, dpi=dpi)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    im = ax.imshow(image, aspect='auto')

    if gt is not None:
        gt_rect = plt.Rectangle(tuple(gt[0, :2]), gt[0, 2]-gt[0, 0], gt[0, 3]-gt[0, 1],
                        linewidth=3, edgecolor="#00ff00", zorder=1, fill=False)
        ax.add_patch(gt_rect)

    for i in range(len(box)):
        rect = plt.Rectangle(tuple(box[i, :2]), box[i, 2]-box[i, 0], box[i, 3]-box[i, 1],
                    linewidth=3, edgecolor="#ff0000", zorder=1, fill=False)
        ax.add_patch(rect)

    if savefig_path is not None:
        fig.savefig(savefig_path, dpi=dpi)
    plt.close()


def run_mdnet(imgs, init_bbox, gt=None, seq='', label_id=0, savefig_dir='', display=False):
    # Init bbox
    target_bbox = np.array(init_bbox[0], dtype='float')
    result = np.zeros((len(imgs), 4))
    result[0] = np.copy(target_bbox)
    iou_result = np.zeros((len(imgs), 1))
    savefig_dir = os.path.join("dump/{}/{}".format(seq, label_id))
    os.makedirs(savefig_dir, exist_ok=True)

    # Init model and optimizer
    mdnet = MDNet()
    mdnet.load_fclayer_weights(opts['model_path'])
    mdnet.set_learnable_params(opts['ft_layers'])
    mdnet.OPN.eval()
    mdnet.cuda()
    gaussian_sampler = SampleGenerator("gaussian", opts["img_size"], 0.1, 1.2)
    backend_sampler = SampleGenerator("gaussian", opts["img_size"], 0.2, 1.2)
    uniform_sampler  = SampleGenerator("uniform",  opts["img_size"],   1,   2, 1.1)

    FPN_RoIAlign = Pooler(output_size=(7, 7),
                          scales=(0.25, 0.125, 0.0625, 0.03125),
                          sampling_ratio=2)
    
    criterion = BinaryLoss()
    roi_scene_size = opts["img_size"] # (w, h)
    init_optimizer = set_optimizer(mdnet, opts['lr_init'])
    update_optimizer = set_optimizer(mdnet, opts['lr_update'])
    pos_feats_all, neg_feats_all = [], []


    tic = time.time()
    # Load first image and finetune
    init_image = imgs[0:1, :]
    init_opn_feat, init_opn_rois = mdnet.forward_OPN(init_image)
    init_roi_feats = FPN_RoIAlign(init_opn_feat, init_opn_rois)

    init_pos_idx, init_neg_idx = sample_pos_neg_idxs(init_bbox, init_opn_rois, 
                                                     fg_thres=opts['overlap_pos_init'][0],
                                                     bg_thres=opts['overlap_neg_init'][1],
                                                     fg_num=opts['n_pos_init'], bg_num=opts['n_neg_init'])
    if init_pos_idx is not None and init_neg_idx is not None:
        init_pos_feats = init_roi_feats[init_pos_idx]
        init_neg_feats = init_roi_feats[init_neg_idx]
        init_pos_rois_visual = torch.cat([x.bbox for x in init_opn_rois])[init_pos_idx]
    else:
        if boxarea(init_bbox[0]) < 50:
            norm_init_bbox = scalebox(init_bbox[0], 5)
        else:
            norm_init_bbox = init_bbox[0]

        init_pos_rois = gen_samples(gaussian_sampler, norm_init_bbox,
                                              opts['n_pos_init'],
                                              overlap_range=opts['overlap_pos_init'])
        init_neg_rois = gen_samples(uniform_sampler, norm_init_bbox,
                                             opts['n_neg_init'],
                                             overlap_range=opts['overlap_neg_init'])
        
        init_pos_rois = [BoxList(torch.from_numpy(init_pos_rois).cuda(), roi_scene_size, mode="xyxy")]
        init_neg_rois = [BoxList(torch.from_numpy(init_neg_rois).cuda(), roi_scene_size, mode="xyxy")]
        init_pos_feats = FPN_RoIAlign(init_opn_feat, init_pos_rois)
        init_neg_feats = FPN_RoIAlign(init_opn_feat, init_neg_rois)
        init_pos_rois_visual = init_pos_rois[0].bbox

    init_pos_feats = init_pos_feats.view(opts['n_pos_init'], -1) 
    init_neg_feats = init_neg_feats.view(opts['n_neg_init'], -1)

    feat_dim = init_pos_feats.size(-1)
    print (feat_dim)

    torch.cuda.empty_cache()
    init_optimizer.zero_grad()
    train(mdnet, criterion, init_optimizer, init_pos_feats, init_neg_feats, opts['maxiter_init'])

    # Memory
    pos_idx = np.asarray(range(init_pos_feats.size(0)))
    np.random.shuffle(pos_idx)
    pos_feats_all = [init_pos_feats.index_select(0, torch.from_numpy(pos_idx[0:opts['n_pos_update']]).cuda())] 
    neg_idx = np.asarray(range(init_neg_feats.size(0)))
    np.random.shuffle(neg_idx)
    neg_feats_all = [init_neg_feats.index_select(0, torch.from_numpy(neg_idx[0:opts['n_neg_update']]).cuda())]

    spf_total = time.time()-tic

    # Visual
    savefig_path = "{}/00000.jpg".format(savefig_dir)
    print("Dump {}...".format(savefig_path))
    #init_rois = torch.cat([x.bbox for x in init_pos_rois])
    overlay_box(depreprocess(imgs[0]), init_pos_rois_visual.cpu().numpy(), gt[0:1], savefig_path=savefig_path)

    for i in range(1, len(imgs)):
        tic = time.time()
        cur_img = imgs[i:i+1, :]
        cur_opn_feat, cur_opn_rois = mdnet.forward_OPN(cur_img)
        cur_roi_feats = FPN_RoIAlign(cur_opn_feat, cur_opn_rois)
        cur_cand_idx = sample_cand_idxs(target_bbox[None, :], torch.cat([x.bbox for x in cur_opn_rois]), size=roi_scene_size, thres=0.2)
        if cur_cand_idx is not None:
            cur_cand_feats = cur_roi_feats[cur_cand_idx].view(cur_cand_idx.size(0), -1)
            cur_cand_rois = torch.cat([x.bbox for x in cur_opn_rois])[cur_cand_idx]
        else:
            backend_rois = gen_samples(backend_sampler, target_bbox, 200, overlap_range=(0, 0.3))
            backend_rois = [BoxList(torch.from_numpy(backend_rois).cuda(), roi_scene_size, mode="xyxy")]
            cur_cand_rois =  torch.cat([x.bbox for x in backend_rois])
            cur_cand_feats = FPN_RoIAlign(cur_opn_feat, backend_rois)
            cur_cand_feats = cur_cand_feats.view(cur_cand_rois.size(0), -1)

        cur_cand_scores = mdnet.forward(cur_cand_feats, in_layer='fc4')
        top_scores, top_idx = cur_cand_scores[:, 1].topk(5)
        top_idx = top_idx.cpu().numpy()
        target_score = top_scores.data.mean().item()

        success = target_score > 0

        # Save result
        if success:
            target_bbox = cur_cand_rois[top_idx].data.cpu().numpy().mean(axis=0)
            print ("success")
        else:
            target_bbox = result[i-1]
            print ("failed")
        result[i] = target_bbox

        # Data collect
        if success:
            cur_pos_idx, cur_neg_idx = sample_pos_neg_idxs_4tensor(target_bbox[None, :], cur_cand_rois, size=roi_scene_size,
                                                                     fg_thres=opts['overlap_pos_update'][0],
                                                                     bg_thres=opts['overlap_neg_update'][1],
                                                                     fg_num=opts['n_pos_update'], 
                                                                     bg_num=opts['n_neg_update'])
            if cur_pos_idx is not None and cur_neg_idx is not None:
                cur_pos_feats = cur_cand_feats[cur_pos_idx].view(opts['n_pos_update'], -1)
                cur_neg_feats = cur_cand_feats[cur_neg_idx].view(opts['n_neg_update'], -1)
                
                pos_feats_all.append(cur_pos_feats)
                neg_feats_all.append(cur_neg_feats)

                if len(pos_feats_all) > opts['n_frames_long']:
                    del pos_feats_all[0]
                if len(neg_feats_all) > opts['n_frames_short']:
                    del neg_feats_all[0]

        # Short term update
        if not success:
            nframes = min(opts['n_frames_short'], len(pos_feats_all))
            pos_data = torch.stack(pos_feats_all[-nframes:], 0).view(-1, feat_dim)
            neg_data = torch.stack(neg_feats_all, 0).view(-1, feat_dim)
            train(mdnet, criterion, update_optimizer, pos_data, neg_data, opts['maxiter_update'])
        # Long term update
        elif i % opts['long_interval'] == 0:
            pos_data = torch.stack(pos_feats_all, 0).view(-1, feat_dim)
            neg_data = torch.stack(neg_feats_all, 0).view(-1, feat_dim)
            train(mdnet, criterion, update_optimizer, pos_data, neg_data, opts['maxiter_update'])

        spf = time.time()-tic
        spf_total += spf

        # Visual
        savefig_path = "{}/{:05d}.jpg".format(savefig_dir, i)
        print("Dump {}...".format(savefig_path))
        #debug_scores, debug_idx = cur_cand_scores[:, 1].topk(10)
        overlay_box(depreprocess(imgs[i]), target_bbox, gt[i:i+1], savefig_path=savefig_path)
    fps = len(imgs) / spf_total

    return result, fps

    

    
if __name__ == "__main__":
    davis = DAVISRegionDataset(opts, data_preprocess=True)
    #imgs, boxes, seq_name, label_id = davis[0]
    #imgs, boxes, seq_name, label_id = davis[2]    # blackswan/1
    #imgs, boxes, seq_name, label_id = davis[19]    # goldfish/2
    #imgs, boxes, seq_name, label_id = davis[20]    # goldfish/2
    length = len(davis)
    #import pdb
    #pdb.set_trace()
    #imgs, boxes, seq_name, label_id = davis[16]    # drift-straight/1
    total_result = {}
    total_fps = []
    for imgs, boxes, seq_name, label_id in davis:
        result, fps = run_mdnet(imgs, boxes[0:1], boxes, seq=seq_name, label_id=label_id)
        total_result["{}-{}".format(seq_name, label_id)] = {"res": result.tolist(), "fps": fps}
        total_fps.append(fps)
    print (np.mean(np.array(total_fps)))
    #with open("result.json", "w") as f:
    #    json.dump(total_result, f)

