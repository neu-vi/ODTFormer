"""
This file is derived from [CLIP](https://github.com/openai/CLIP/blob/main/clip/model.py).
Modified for [ODTFormer] by Tianye Ding
"""
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


def generate_sampling_offsets(roi_size):
    if isinstance(roi_size, int):
        roi_size = (roi_size, roi_size, roi_size)

    for _ in roi_size:
        assert _ % 2 != 0, 'roi_size must be odd for an explicit center voxel'
    center = [(_ - 1) // 2 for _ in roi_size]
    i, j, k = torch.meshgrid(torch.arange(roi_size[0]), torch.arange(roi_size[1]), torch.arange(roi_size[2]),
                             indexing='ij')

    offsets = torch.stack((i - center[0], j - center[1], k - center[2]), dim=-1)
    return offsets.view(-1, 3)


class VoxelTracker(nn.Module):
    def __init__(self, roi_size):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.offset_vectors = generate_sampling_offsets(roi_size).view(-1, 3)

    def forward(self, voxel_feat_0, voxel_feat_1, spatial_locations, training, src_mask=None, tgt_mask=None):
        """

        @param voxel_feat_0:
        @param voxel_feat_1:
        @param spatial_locations: X, Y, Z, 3, starting at (0, 0, 0)
        @param training:
        @param src_mask:
        @param tgt_mask:
        @return:
        """
        N, C, X, Y, Z = voxel_feat_0.shape
        spatial_locations = spatial_locations[None, ...].expand(N, -1, -1)
        empty_volume = torch.zeros_like(spatial_locations)

        # normalize features
        feat_0_norm = voxel_feat_0 / (torch.linalg.norm(voxel_feat_0, axis=1, keepdim=True) + 1e-7)
        feat_1_norm = voxel_feat_1 / (torch.linalg.norm(voxel_feat_1, axis=1, keepdim=True) + 1e-7)

        occupied_mask = torch.ones(spatial_locations.shape[:-1], device=spatial_locations.device)
        if src_mask is not None:
            src_mask = src_mask.view(N, -1)
            assert src_mask.shape == spatial_locations.shape[:-1]
            occupied_mask = src_mask
        occupied_mask = occupied_mask.view(N, -1)
        occupied_voxels = int(max(torch.sum(occupied_mask, dim=-1)).item())
        # N, num_max_occupied_voxels (batch)
        topk_idx = torch.topk(occupied_mask, occupied_voxels, dim=1)[1]
        # N, num_occupied_voxels, 3
        gather_idx = topk_idx[:, :, None].expand(-1, -1, 3)

        # N, num_occupied_voxels, 3
        spatial_locations = spatial_locations.gather(1, gather_idx)

        # [N, num_occupied_voxels], roi_num_voxels, 3
        offset_field = self.offset_vectors.to(feat_0_norm.device, dtype=empty_volume.dtype)[None, None, ...].expand(
            N, occupied_voxels, -1, -1)
        # N, num_occupied_voxels, roi_num_voxels, 3
        sampling_locations = spatial_locations[..., None, :] + offset_field

        selected_voxels = []
        if tgt_mask is not None:
            # N, [num_occupied_voxels], tgt_num_voxels, 3
            tgt_sampling = torch.nonzero(tgt_mask)[:, 1:][None, None, ...].expand(N, occupied_voxels, -1, -1)
            tgt_sampling = torch.concat([tgt_sampling, spatial_locations[..., None, :]], dim=-2)

            # N, num_occupied_voxels, roi_num_voxels, (tgt_num_voxels), (3)
            selected_voxels = (tgt_sampling[..., None, :, :] == sampling_locations[..., None, :]).all(-1).any(-1)

        sampling_locations_norm = 2 * (
                sampling_locations / (torch.tensor([X, Y, Z], device=sampling_locations.device) - 1)) - 1
        # N, 1, num_occupied_voxels, roi_num_voxels, 3 [D, H, W index -> X, Y, Z coord]
        sampling_grid = sampling_locations_norm[:, None, ...].flip(-1)
        # N, C, num_occupied_voxels, roi_num_voxels
        sample_feat_1 = F.grid_sample(feat_1_norm, sampling_grid, mode='nearest', padding_mode='zeros',
                                      align_corners=True).squeeze(2)

        feat_0_flatten = feat_0_norm.view(N, C, -1).gather(-1, topk_idx[:, None, :].expand(-1, C, -1))
        # cosine similarity as logit
        logit_scale = self.logit_scale.exp()
        # N, num_occupied_voxel, roi_num_voxels
        logit_0 = (feat_0_flatten.transpose(1, 2).unsqueeze(2) @ sample_feat_1.transpose(1, 2)).squeeze(2)
        logit_voxel_0 = logit_scale * logit_0
        if tgt_mask is not None:
            logit_voxel_0[torch.bitwise_not(selected_voxels)] = -1.

        prob_voxel_0 = F.softmax(logit_voxel_0, dim=-1)
        if training:
            # N, num_occupied_voxel, roi_num_voxels, 1 -> N, num_occupied_voxel, (roi_num_voxels), 3
            flow_est_3d = torch.sum(prob_voxel_0.unsqueeze(-1) * offset_field, dim=-2)
        else:
            # N, num_occupied_voxel, (1), 3
            max_prob_flow_3d = torch.argmax(prob_voxel_0, dim=-1, keepdim=True)[..., None].expand(-1, -1, -1, 3)
            flow_est_3d = offset_field.gather(-2, max_prob_flow_3d).squeeze(-2)
        # N, num_occupied_voxel, 3 -> N, X, Y, Z, 3
        flow_volume = empty_volume.scatter(1, gather_idx, flow_est_3d)

        return flow_volume.view(N, X, Y, Z, 3)
