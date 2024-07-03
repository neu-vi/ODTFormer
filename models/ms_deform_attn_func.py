# This file is derived from [Deformable-DETR](https://github.com/fundamentalvision/Deformable-DETR/blob/main/models/ops/functions/ms_deform_attn_func.py).
# Modified for [ODTFormer] by Tianye Ding
#
# Original header:
# ------------------------------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------------------
# Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
# ------------------------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import torch
import torch.nn.functional as F


def ms_deform_attn_core_pytorch(value, value_spatial_shapes, sampling_locations, matching_layer, feature_distillation,
                                attention_weights):
    # for debug and test only,
    # need to use cuda version instead
    N_, S_, M_, D_ = value.shape
    _, Lq_, M_, L_, P_, _ = sampling_locations.shape
    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)
    sampling_grids = 2 * sampling_locations - 1
    assert P_ % 2 == 0, 'only sampled from 1 image'
    num_points = P_ // 2

    sampling_value_list = []
    for lid_, (H_, W_) in enumerate(value_spatial_shapes):
        # N_, H_*W_, M_, D_ -> N_, H_*W_, M_*D_ -> N_, M_*D_, H_*W_ -> N_*M_, D_, H_, W_
        value_l_ = value_list[lid_].flatten(2).transpose(1, 2).reshape(N_ * M_, D_, H_, W_)
        # N_, Lq_, M_, P_, 2 -> N_, M_, Lq_, P_, 2 -> N_*M_, Lq_, P_, 2
        sampling_grid_l_ = sampling_grids[:, :, :, lid_].transpose(1, 2).flatten(0, 1)
        # N_*M_, D_, Lq_, P_
        sampling_value_l_ = F.grid_sample(value_l_, sampling_grid_l_,
                                          mode='bilinear', padding_mode='zeros', align_corners=False)
        # ========== Matching Cost ==========
        # N_*M_, 2 * D_, Lq_, num_points
        cost_volume_l_ = torch.concat([sampling_value_l_[..., :num_points], sampling_value_l_[..., num_points:]], dim=1)

        cost_volume_l_ = cost_volume_l_.permute(0, 2, 3, 1).contiguous()
        # N_*M_, Lq_, num_points, 2 * D_ -> N_*M_, Lq_, num_points, D_
        volume_l_ = matching_layer[lid_](cost_volume_l_.view(-1, 2 * D_)).reshape(N_ * M_, Lq_, num_points, D_)
        # N_ * M_, D_, Lq_, num_points
        # volume_l_ = volume_l_.permute(0, 3, 1, 2).contiguous()
        # ===================================
        sampling_value_list.append(volume_l_)

    # ===================================
    # Level Feature Distillation
    # ===================================
    # (N_, M_), Lq_, num_points, (L_, D_)
    sampling_value_ = torch.stack(sampling_value_list, dim=-2).view(N_ * M_ * Lq_ * num_points, -1)
    # (N_, M_), Lq_, num_points, D_ -> (N_, M_), D_, Lq_, num_points
    sampling_value = feature_distillation(sampling_value_).view(N_ * M_, Lq_, num_points, D_).permute(0, 3, 1,
                                                                                                      2).contiguous()

    # N_, Lq_, M_, P_ -> N_, M_, Lq_, P_ -> (N_ * M_), 1, Lq_, P_
    attention_weights = attention_weights.transpose(1, 2).reshape(N_ * M_, 1, Lq_, num_points)
    output = (sampling_value * attention_weights).sum(-1).view(N_, M_ * D_, Lq_)
    return output.transpose(1, 2).contiguous()
