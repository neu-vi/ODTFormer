"""
This file is derived from [Deformable-DETR](https://github.com/fundamentalvision/Deformable-DETR/blob/main/models/deformable_transformer.py).
Modified for [ODTFormer] by Tianye Ding

Original header:
------------------------------------------------------------------------
Deformable DETR
Copyright (c) 2020 SenseTime. All Rights Reserved.
Licensed under the Apache License, Version 2.0 [see LICENSE for details]
------------------------------------------------------------------------
Modified from DETR (https://github.com/facebookresearch/detr)
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
------------------------------------------------------------------------
"""
import copy
from typing import Optional

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import Tensor
from torch.nn.functional import grid_sample
from torch.nn.init import normal_

# local
from .mv_deform_attn import VoxDeformCrossAttn


def project(memory_hw, query_pos, T_camera_local, camera, ooi_embed):
    '''
    Project reference points onto multi-view image to fetch appearence features
    Bilinear interpolation is used to fetch features.
    Average pooling is used to aggregate features from different views.
    '''
    w, h = camera.size[0][0].data.cpu().numpy()
    # from local coord to camera coord
    # B x T x L x 3
    query_pos_c = T_camera_local.transform(query_pos.unsqueeze(1))

    center_b_list = []
    valid_b_list = []
    for cam_b, pos_c_b in zip(camera, query_pos_c):
        center_im_b, center_valid_b = cam_b.project(pos_c_b)
        center_b_list.append(center_im_b)
        valid_b_list.append(center_valid_b)
    # B x T x L x 2
    center_im = torch.stack(center_b_list)
    # B x T x L
    center_valid = torch.stack(valid_b_list)

    im_grid = torch.stack([2 * center_im[..., 0] / (w - 1) - 1, 2 * center_im[..., 1] / (h - 1) - 1], dim=-1)
    bs, num_view, num_query, _ = im_grid.shape
    # (B*T) x 1 x L x 2
    im_grid = im_grid.view(bs * num_view, 1, -1, 2)

    # (B*T) x C x 1 x L
    features = grid_sample(memory_hw, im_grid, padding_mode='zeros', align_corners=True)

    if ooi_embed is not None:
        # add out of image embedding
        # (B*T) x L
        center_valid_reshape = center_valid.view(bs * num_view, -1)
        center_valid_reshape = center_valid_reshape.unsqueeze(1).unsqueeze(2).float()
        features = features * center_valid_reshape + ooi_embed * (1 - center_valid_reshape)

    features = features.view(bs, num_view, -1, num_query)
    features = features.permute(0, 1, 3, 2).contiguous()
    # average across different views
    features = features.sum(dim=1)
    mask = center_valid.sum(dim=1)
    invalid_mask = mask == 0
    mask[invalid_mask] = 1
    if ooi_embed is not None:
        features /= num_view
    else:
        features /= mask.unsqueeze(-1)
    return features, center_im, center_valid


class DeformableTransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, dim_in, scale, norm=None, return_intermediate=False,
                 share_weights=False, use_ooi_embed=False):
        super().__init__()
        if not share_weights:
            self.layers = _get_clones(decoder_layer, num_layers)
        else:
            self.layers = _get_clones(decoder_layer, 1)
        self.num_layers = num_layers
        self.share_weights = share_weights
        self.norm = norm
        self.return_intermediate = return_intermediate
        self.mlp_heads = None
        self.box_processor = None
        self.scale = scale

        if use_ooi_embed:
            self.ooi_embed = nn.Parameter(torch.zeros((1, dim_in, 1, 1)), requires_grad=True)
            normal_(self.ooi_embed)
        else:
            self.ooi_embed = None

    def normalize(self, center_offset):
        center_offset1 = (center_offset[..., 0] - self.scale[0]) / (
                self.scale[1] - self.scale[0]
        )
        center_offset2 = (center_offset[..., 1] - self.scale[2]) / (
                self.scale[3] - self.scale[2]
        )
        center_offset3 = (center_offset[..., 2] - self.scale[4]) / (
                self.scale[5] - self.scale[4]
        )
        center_offset = torch.stack([center_offset1, center_offset2, center_offset3], dim=-1)
        return center_offset

    def denormalize(self, center_offset):
        center_offset1 = (
                center_offset[..., 0] * (self.scale[1] - self.scale[0]) + self.scale[0]
        )
        center_offset2 = (
                center_offset[..., 1] * (self.scale[3] - self.scale[2]) + self.scale[2]
        )
        center_offset3 = (
                center_offset[..., 2] * (self.scale[5] - self.scale[4]) + self.scale[4]
        )
        center_offset = torch.stack([center_offset1, center_offset2, center_offset3], dim=-1)
        return center_offset

    def forward(self,
                pos_feat,
                memory,
                spatial_shape,
                *,
                level_start_index: Optional[Tensor] = None,
                reference_points: Optional[Tensor] = None,
                ms_reference_points: Optional[Tensor] = None,
                feature_level: Optional[int] = None,
                multiscale_cam,
                meta_data,
                sample_size):
        # parse meta data
        camera = meta_data["camera"]
        T_camera_pseudoCam = meta_data["T_camera_pseudoCam"]
        T_world_pseudoCam = meta_data["T_world_pseudoCam"]
        T_world_local = meta_data["T_world_local"]
        bs, num_view = T_camera_pseudoCam.shape[:2]

        T_camera_local = T_camera_pseudoCam @ (
                T_world_pseudoCam.inverse() @ T_world_local
        )

        if feature_level is not None:
            h, w = spatial_shape[feature_level].cpu().tolist()
            ##############################################################
            start_index = level_start_index[feature_level]
            memory_hw = memory[..., start_index:start_index + int(h) * int(w), :].reshape(bs * num_view,
                                                                                          int(h / num_view), int(w), -1)
            memory_hw = memory_hw.permute(0, 3, 1, 2)

            # project
            pixel_aligned, center_im, center_valid = project(memory_hw, self.denormalize(reference_points),
                                                             T_camera_local, camera, self.ooi_embed)
            query_pos = pixel_aligned
        else:
            query_pos = None

        output = pos_feat

        for layer_num in range(self.num_layers):
            if self.share_weights:
                layer = self.layers[0]
            else:
                layer = self.layers[layer_num]

            output = layer(output, memory, spatial_shape,
                           query_pos=query_pos,
                           multiscale_cam=multiscale_cam,
                           reference_points=ms_reference_points,
                           sample_size=sample_size)

        return output


class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, ff_dim, num_cams, num_levels, num_points=4, dropout=0.1,
                 **kwargs):
        super().__init__()

        # cross attention
        self.cross_attn = VoxDeformCrossAttn(d_model, num_cams, num_heads, num_levels, num_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, ff_dim)
        self.activation = nn.ReLU(inplace=True)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(ff_dim, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(self, tgt, memory, spatial_shape, query_pos, multiscale_cam, reference_points, sample_size):
        # self attention
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, tgt)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # cross attention
        tgt2 = self.cross_attn(tgt, memory, spatial_shape,
                               query_pos=query_pos,
                               multiscale_cam=multiscale_cam,
                               reference_points=reference_points,
                               sample_size=sample_size)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ffn
        tgt = self.forward_ffn(tgt)

        return tgt


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
