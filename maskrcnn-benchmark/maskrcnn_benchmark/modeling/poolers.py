# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torch.nn.functional as F
from torch import nn

#import sys
#sys.path.insert(0, "../..")

from maskrcnn_benchmark.layers import ROIAlign

from .utils import cat
#from utils import cat


class LevelMapper(object):
    """Determine which FPN level each RoI in a set of RoIs should map to based
    on the heuristic in the FPN paper.
    """

    def __init__(self, k_min, k_max, canonical_scale=224, canonical_level=4, eps=1e-6):
        """
        Arguments:
            k_min (int)
            k_max (int)
            canonical_scale (int)
            canonical_level (int)
            eps (float)
        """
        self.k_min = k_min
        self.k_max = k_max
        self.s0 = canonical_scale
        self.lvl0 = canonical_level
        self.eps = eps

    def __call__(self, boxlists):
        """
        Arguments:
            boxlists (list[BoxList])
        """
        # Compute level ids
        s = torch.sqrt(cat([boxlist.area() for boxlist in boxlists]))

        # Eqn.(1) in FPN paper
        target_lvls = torch.floor(self.lvl0 + torch.log2(s / self.s0 + self.eps))
        target_lvls = torch.clamp(target_lvls, min=self.k_min, max=self.k_max)
        return target_lvls.to(torch.int64) - self.k_min


class Pooler(nn.Module):
    """
    Pooler for Detection with or without FPN.
    It currently hard-code ROIAlign in the implementation,
    but that can be made more generic later on.
    Also, the requirement of passing the scales is not strictly necessary, as they
    can be inferred from the size of the feature map / size of original image,
    which is available thanks to the BoxList.
    """

    def __init__(self, output_size, scales, sampling_ratio):
        """
        Arguments:
            output_size (list[tuple[int]] or list[int]): output size for the pooled region
            scales (list[float]): scales for each Pooler
            sampling_ratio (int): sampling ratio for ROIAlign
        """
        super(Pooler, self).__init__()
        poolers = []
        for scale in scales:
            poolers.append(
                ROIAlign(
                    output_size, spatial_scale=scale, sampling_ratio=sampling_ratio
                )
            )
        self.poolers = nn.ModuleList(poolers)
        self.output_size = output_size
        # get the levels in the feature map by leveraging the fact that the network always
        # downsamples by a factor of 2 at each level.
        lvl_min = -torch.log2(torch.tensor(scales[0], dtype=torch.float32)).item()
        lvl_max = -torch.log2(torch.tensor(scales[-1], dtype=torch.float32)).item()
        self.map_levels = LevelMapper(lvl_min, lvl_max)

    def convert_to_roi_format(self, boxes):
        concat_boxes = cat([b.bbox for b in boxes], dim=0)
        device, dtype = concat_boxes.device, concat_boxes.dtype
        ids = cat(
            [
                torch.full((len(b), 1), i, dtype=dtype, device=device)
                for i, b in enumerate(boxes)
            ],
            dim=0,
        )
        rois = torch.cat([ids, concat_boxes], dim=1)
        return rois

    def forward(self, x, boxes):
        import pdb
        pdb.set_trace()
        """
        Arguments:
            x (list[Tensor]): feature maps for each level
            boxes (list[BoxList]): boxes to be used to perform the pooling operation.
        Returns:
            result (Tensor)
        """
        num_levels = len(self.poolers)
        rois = self.convert_to_roi_format(boxes)
        if num_levels == 1:
            return self.poolers[0](x[0], rois)

        levels = self.map_levels(boxes)

        num_rois = len(rois)
        num_channels = x[0].shape[1]
        output_size = self.output_size[0]

        dtype, device = x[0].dtype, x[0].device
        result = torch.zeros(
            (num_rois, num_channels, output_size, output_size),
            dtype=dtype,
            device=device,
        )
        for level, (per_level_feature, pooler) in enumerate(zip(x, self.poolers)):
            idx_in_level = torch.nonzero(levels == level).squeeze(1)
            rois_per_level = rois[idx_in_level]
            result[idx_in_level] = pooler(per_level_feature, rois_per_level)

        return result

class Encoder_Pooler(nn.Module):
    """
    Encoder_Pooler for Decoder, each scale will be pooling to a fixed size.
    In RGMP, (1/4, 1/8, 1/16) are used to refine final results, and input is (256, 256).
    In PTSNet, we use (1/4, 1/8, 1/16) also to refine final results. The feature map
      to be RoIAligned are (120x216, 60x108, 30x54). The pooling feature sizes are
      (64x64. 32x32, 16x16).
    Also, the requirement of passing the scales is not strictly necessary, as they
    can be inferred from the size of the feature map / size of original image,
    which is available thanks to the BoxList.
    """

    def __init__(self, output_sizes, scales, sampling_ratio):
        """
        Arguments:
            output_size (list[tuple[int]] or list[int]): output size for the pooled region
            scales (list[float]): scales for each Pooler
            sampling_ratio (int): sampling ratio for ROIAlign
        """
        super(Encoder_Pooler, self).__init__()
        assert len(output_sizes) == len(scales)
        poolers = []
        for output_size, scale in zip(output_sizes, scales):
            poolers.append(
                ROIAlign(
                    (output_size, output_size), spatial_scale=scale, 
                    sampling_ratio=sampling_ratio
                )
            )
        self.poolers = nn.ModuleList(poolers)
        self.output_sizes = output_sizes

    def convert_to_roi_format(self, boxes):
        concat_boxes = cat([b.bbox for b in boxes], dim=0)
        device, dtype = concat_boxes.device, concat_boxes.dtype
        ids = cat(
            [
                torch.full((len(b), 1), i, dtype=dtype, device=device)
                for i, b in enumerate(boxes)
            ],
            dim=0,
        )
        rois = torch.cat([ids, concat_boxes], dim=1)
        return rois

    def forward(self, x, boxes, convert=False):
        """
        Arguments:
            x ([Nx256xHxW, Nx256xH'xW' ...]): feature maps for each level
            boxes (Nx4 np array): rois
        Returns:
            result (list[Tensor])
        """
        assert boxes.dim() == 2
        dtype, device = x[0].dtype, x[0].device
        rois = boxes.type(dtype).to(device)
        num_rois = len(rois)
        #if boxes.dim() == 1: boxes = boxes[None, :]
        # RoIs is a Nx5 Tensor. Note the arange here, it is the id of which channel.
        if rois.size(1) == 4: 
            rois = torch.cat(
                        [torch.arange(num_rois, dtype=dtype, device=device).unsqueeze(1), rois], 
                        dim=1)
        result = []
        for per_level_feature, pooler in zip(x, self.poolers):
            per_aligned_feature = pooler(per_level_feature, rois)
            result.append(per_aligned_feature)
        return result

if __name__ == "__main__":
    p = Encoder_Pooler([10, 5], [0.5, 0.25], 2)
    aa = [torch.zeros((2, 256, 100, 100)), torch.ones((2, 256, 50, 50))]
    aa[0][1] = 1
    bb = torch.Tensor([[1, 1, 10, 10], [2, 2, 8, 8]])
    xx = p(aa, bb)
    import pdb
    pdb.set_trace()

