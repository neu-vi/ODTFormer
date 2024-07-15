# This file is derived from [Deformable-DETR](https://github.com/fundamentalvision/Deformable-DETR/blob/main/models/ops/modules/ms_deform_attn.py).
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
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_, normal_

# local
from .ms_deform_attn_func import ms_deform_attn_core_pytorch


class VoxDeformCrossAttn(nn.Module):
    def __init__(self, d_model, num_cams, num_heads, num_levels, num_points, dropout=0.1):
        """

        @param d_model:
        @param num_cams:
        @param num_heads:
        @param num_points:
        @param dropout:
        """
        super().__init__()
        self.deform_attn = MVDeformAttn(d_model, num_heads, num_levels, num_cams, num_points)
        self.offset_sampler = OffsetSampler(d_model, num_heads, 1, num_points)

        self.d_model = d_model
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_cams = num_cams
        self.num_points = num_points

        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, memory, spatial_shapes,
                *,
                query_pos=None, multiscale_cam, reference_points, sample_size):
        """

        @param tgt: N, num_query, C
        @param memory: N, Lx(TxH)xW, C
        @param spatial_shapes: TxH, W
        @param query_pos:
        @param multiscale_cam: N, L, T, 6
        @param reference_points: N, num_query, 3 - x, y
        @param sample_size: sample bbox size in real-world metric
        @return:
        """
        bs, num_query, _ = tgt.shape
        inp_residual = tgt

        # sampling offsets in 3D space
        sampling_offsets = self.offset_sampler(tgt).view(bs, num_query, self.num_heads, self.num_points, 3)

        sample_size = torch.tensor([sample_size, sample_size, sample_size], device=reference_points.device)
        denorm_offsets = -sample_size / 2 + sampling_offsets * sample_size

        if reference_points.shape[-1] == 3:
            # B, Lq, 3 -> B, Lq, N, num_points, 3
            sampling_locations = reference_points[..., None, None, :] + denorm_offsets
            # B, Lq, N, num_points, 3 -> B, (Lq, N, num_points), 3
            sampling_locations = sampling_locations.reshape(bs, -1, 3)
        else:
            raise ValueError(
                'Last dim of reference_points must be 3, but get {} instead.'.format(reference_points.shape[-1]))

        # project 3D sampling points back to 2D
        # B, (Lq, N, num_points), 3 -> B, L, T, (Lq, N, num_points), 3
        sampling_locations = sampling_locations[:, None, None, :, :].expand(-1, self.num_levels, self.num_cams, -1, -1)
        # B, L, T, (Lq, N, num_points), 3 -> B, L, T, Lq, N, num_points, 2
        proj_sampling_locations = multiscale_cam.project(sampling_locations)[0].view(bs, self.num_levels, self.num_cams,
                                                                                     num_query, self.num_heads,
                                                                                     self.num_points, 2)
        # B, L, T, Lq, N, num_points, 2 -> B, Lq, N, L, T, num_points, 2
        proj_sampling_locations = proj_sampling_locations.permute(0, 3, 4, 1, 2, 5, 6).contiguous()

        if query_pos is not None:
            tgt = tgt + query_pos

        key = value = memory
        queries = self.deform_attn(tgt,
                                   key,
                                   value,
                                   spatial_shapes,
                                   proj_sampling_locations)

        return self.dropout(queries) + inp_residual


class MVDeformAttn(nn.Module):
    def __init__(self, d_model, num_heads, num_levels, num_cams, num_points, batch_first=True):
        """

        @param d_model:
        @param num_heads:
        @param num_levels
        @param num_cams:
        @param num_points:
        @param batch_first:
        """
        super().__init__()
        assert d_model % num_heads == 0, f'd_model must be divisible by num_heads, but got {d_model} and {num_heads}'
        dim_per_head = d_model // num_heads

        self.d_model = d_model
        self.num_cams = num_cams
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.dim_per_head = dim_per_head

        self.attention_weights = nn.Linear(d_model, self.num_heads * self.num_points)
        self.matching_layer = nn.ModuleList([nn.Sequential(
            nn.Linear(2 * dim_per_head, 2 * dim_per_head, bias=False),
            nn.BatchNorm1d(2 * dim_per_head),
            nn.ReLU(),
            nn.Linear(2 * dim_per_head, dim_per_head, bias=False),
            nn.BatchNorm1d(dim_per_head),
            nn.ReLU()) for _ in range(num_levels)])
        self.feature_distillation = nn.Sequential(
            nn.Linear(num_levels * dim_per_head, num_levels * dim_per_head, bias=False),
            nn.BatchNorm1d(num_levels * dim_per_head),
            nn.ReLU(),
            nn.Linear(num_levels * dim_per_head, dim_per_head, bias=False),
            nn.BatchNorm1d(dim_per_head),
            nn.ReLU()
        )

        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        self.batch_first = batch_first

        self._reset_parameters()

    def _reset_parameters(self):
        constant_(self.attention_weights.weight.data, 0.)
        constant_(self.attention_weights.bias.data, 0.)

        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

    def forward(self, query, key, value, spatial_shapes, sampling_locations):
        """

        @param query: N, num_query, C
        @param key: N, Lx(TxH)xW, C
        @param value:
        @param spatial_shapes: 2xH, W
        @param sampling_locations: B, Lq, N, L, T, num_points, 2 - x, y
        @return:
        """
        if not self.batch_first:
            # (N, B, C) -> (B, N ,C)
            query = query.permute(1, 0, 2).contiguous()
            key = key.permute(1, 0, 2).contiguous()

        if value is None:
            value = key

        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape

        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

        value = self.value_proj(value)
        value = value.view(bs, num_value, self.num_heads, self.dim_per_head)

        # H, W -> X, Y
        view_spatial_shapes = torch.flip(spatial_shapes, dims=(-1,))
        # L, 2
        view_spatial_shapes[..., 1] //= self.num_cams

        attention_weights = self.attention_weights(query).view(bs, num_query, self.num_heads, self.num_points)
        attention_weights = F.softmax(attention_weights, -1).view(bs, num_query, self.num_heads, self.num_points)

        sampling_locations /= view_spatial_shapes[None, None, None, :, None, None, :]
        output = ms_deform_attn_core_pytorch(value, spatial_shapes, self.num_cams, sampling_locations,
                                             self.matching_layer, self.feature_distillation, attention_weights)

        output = self.output_proj(output)
        if not self.batch_first:
            output = output.permute(1, 0, 2)

        return output


class OffsetSampler(nn.Module):
    def __init__(self, embed_dim, num_heads, num_cams, num_points):
        super().__init__()
        self.offset_sampler = nn.Linear(embed_dim, num_heads * num_cams * num_points * 3)
        self.norm = nn.Sigmoid()

        self.num_heads = num_heads
        self.num_cams = num_cams
        self.num_points = num_points

        self._reset_parameters()

    def _reset_parameters(self):
        constant_(self.offset_sampler.weight.data, 0)
        with torch.no_grad():
            self.offset_sampler.bias = nn.Parameter(
                torch.zeros([self.num_heads, self.num_cams, self.num_points, 3]).view(-1))
            normal_(self.offset_sampler.bias, mean=0., std=1.)

    def forward(self, query):
        sample_offsets = self.offset_sampler(query)
        norm = self.norm(sample_offsets.contiguous())
        return norm
