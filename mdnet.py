#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2019 qiang.zhou <theodoruszq@gmail.com>

"""
MDNet coupled with OPN.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

import sys
sys.path.append('./maskrcnn-benchmark')
import maskrcnn_fea_and_prop

def append_params(params, module, prefix):
    for child in module.children():
        for k, p in child._parameters.items():
            if p is None: continue

            if isinstance(child, nn.BatchNorm2d):
                name = prefix + '_bn_' + k
            else:
                name = prefix + '_' + k

            if name not in params:
                params[name] = p
            else:
                raise RuntimeError("Duplicated param name: %s" % (name))

class MDNet(nn.Module):
    def __init__(self, K=1):
        super(MDNet, self).__init__()
        self.K = K
        
        self.OPN = maskrcnn_fea_and_prop.get_opn()
        self.fclayers = nn.Sequential(OrderedDict([
                            ('fc4', nn.Sequential(nn.Linear(256 * 7 * 7, 512), 
                                                  nn.ReLU())),
                            ('fc5', nn.Sequential(nn.Dropout(0.5), 
                                                  nn.Linear(512, 512), 
                                                  nn.ReLU())),
                            ]))
            
        self.branches = nn.ModuleList([nn.Sequential(nn.Dropout(0.5),
                                                     nn.Linear(512, 2)) for _ in range(K)])
        
        # self.roi_align_model = RoIAlignMax(3, 3, 1./8)
        self.receptive_field = 75.
        self.build_param_dict()
    
    def load_fclayer_weights(self, model_path):
        fclayers_state_dict = torch.load(model_path)['fclayers']
        self.fclayers.load_state_dict(fclayers_state_dict)

    def build_param_dict(self):
        self.params = OrderedDict()
        for name, module in self.OPN.named_children():
            append_params(self.params, module, name)
        for name, module in self.fclayers.named_children():
            append_params(self.params, module, name)
        for k, module in enumerate(self.branches):
            append_params(self.params, module, 'fc6_%d'%(k))

    def set_learnable_params(self, layers):
        for k, p in self.params.items():
            if any([k.startswith(l) for l in layers]):
                p.requires_grad = True
            else:
                p.requires_grad = False

    def get_learnable_params(self):
        params = OrderedDict()
        for k, p in self.params.items():
            if p.requires_grad:
                params[k] = p
        return params
    
    def forward_OPN(self, images):
        opn_feat, opn_rois = self.OPN(images)
        return opn_feat, opn_rois

    def forward(self, x, k=0, in_layer='opn', out_layer='fc6'):
        if in_layer == "opn" and out_layer == 'fc4':
            features, proposals = self.OPN(x)
        elif in_layer == 'opn' and out_layer == 'fc6':
            features, proposals = self.OPN(x)
            x = self.fclayers(x)
        elif in_layer == 'fc4' and out_layer == 'fc6':
            x = self.fclayers(x)
        else:
            assert False, "in_layer and out_layer are not reasonable"

        x = self.branches[k](x)
        if out_layer == 'fc6':
            return x
        elif out_layer == "fc6_softmax":
            return F.softmax(x)


class BinaryLoss(nn.Module):
    def __init__(self):
        super(BinaryLoss, self).__init__()

    def forward(self, pos_score, neg_score):
        pos_loss = -F.log_softmax(pos_score, dim=1)[:,1]
        neg_loss = -F.log_softmax(neg_score, dim=1)[:,0]

        loss = (pos_loss.sum() + neg_loss.sum())/(pos_loss.size(0) + neg_loss.size(0))
        return loss


class Accuracy():
    def __call__(self, pos_score, neg_score):

        pos_correct = (pos_score[:,1] > pos_score[:,0]).sum().float()
        neg_correct = (neg_score[:,1] < neg_score[:,0]).sum().float()

        pos_acc = pos_correct / (pos_score.size(0) + 1e-8)
        neg_acc = neg_correct / (neg_score.size(0) + 1e-8)

        return pos_acc.data[0], neg_acc.data[0]

class Precision():
    def __call__(self, pos_score, neg_score):

        scores = torch.cat((pos_score[:, 1], neg_score[:, 1]), 0)
        topk = torch.topk(scores, pos_score.size(0))[1]
        prec = (topk < pos_score.size(0)).float().sum() / (pos_score.size(0)+1e-8)

        return prec.item()



if __name__ == "__main__":
    mdnet = MDNet()

    mdnet = mdnet.cuda()
    mdnet.load_fclayer_weights("snapshots/rt_mdnet_fclayers.pth")   
    import numpy as np
    #aa = np.ones((8, 3, 107, 107), dtype=np.uint8)
    #bb = mdnet.OPN(aa, in_layer='opn', out_layer='fc6')
    import pdb
    pdb.set_trace()
