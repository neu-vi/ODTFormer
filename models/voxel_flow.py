# global
import numpy as np
from torch import nn
import hydra
from tqdm import tqdm
import os

# local
from .tracker import VoxelTracker
from .odtformer import ref_points_generator


class VoxelFlow(nn.Module):
    def __init__(self, voxel_net, voxel_grid_scale, voxel_size, roi_size):
        super().__init__()
        self.voxel_net = hydra.utils.instantiate(voxel_net, _recursive_=False)
        self.track_head = VoxelTracker(roi_size)

        self.voxel_size = voxel_size
        min_x, max_x, min_y, max_y, min_z, max_z = voxel_grid_scale
        grid_shape = [int((max_x - min_x) // voxel_size),
                      int((max_y - min_y) // voxel_size),
                      int((max_z - min_z) // voxel_size)]
        self.spatial_locations = ref_points_generator([-.5, -.5, -.5], grid_shape, 1, normalize=False).view(-1, 3)

    def forward(self, img_pair_0, img_pair_1,
                *,
                training=True, pair_0_kwargs, pair_1_kwargs, src_mask=None):
        out_0, voxel_0 = self.voxel_net(*img_pair_0, **pair_0_kwargs)
        out_1, voxel_1 = self.voxel_net(*img_pair_1, **pair_1_kwargs)

        if src_mask is None:
            src_mask = out_0[-1].clone().detach()
            src_mask[src_mask < 0.5] = 0
            src_mask[src_mask >= 0.5] = 1
        if training:
            unit_flow = self.track_head(voxel_0, voxel_1, self.spatial_locations.to(voxel_0.device), training=True,
                                        src_mask=src_mask)
        else:
            tgt_mask = out_1[-1].clone().detach()
            tgt_mask[tgt_mask < 0.5] = 0
            tgt_mask[tgt_mask >= 0.5] = 1
            unit_flow = self.track_head(voxel_0, voxel_1, self.spatial_locations.to(voxel_0.device), training=False,
                                        src_mask=src_mask)  # tgt_mask=tgt_mask)
        return (out_0, out_1), unit_flow * self.voxel_size

    def seq_forward(self, img_pair_seq, pair_seq_kwargs, *, savedir):
        assert len(img_pair_seq) == len(pair_seq_kwargs)
        voxel_grid_, voxel_feat = self.voxel_net(*img_pair_seq[0], training=False, **pair_seq_kwargs[0])

        spatial_loc = self.spatial_locations.to(voxel_feat.device)
        for voxel_pred in voxel_grid_:
            voxel_pred[voxel_pred < 0.5] = 0
            voxel_pred[voxel_pred >= 0.5] = 1

        # torch.save(voxel_grid_[-1].cpu(), os.path.join(savedir, 'vox_grid_seq_0.pt'))
        np.save(os.path.join(savedir, 'vox_grid_seq_0.npy'), voxel_grid_[-1].cpu().numpy())
        voxel_grid_out = voxel_grid_

        for idx, img_pair in enumerate(tqdm(img_pair_seq[1:])):
            offset_idx = idx + 1
            voxel_grid_1, voxel_feat_1 = self.voxel_net(*img_pair, training=False, **pair_seq_kwargs[offset_idx])
            for voxel_pred in voxel_grid_1:
                voxel_pred[voxel_pred < 0.5] = 0
                voxel_pred[voxel_pred >= 0.5] = 1
            unit_flow = self.track_head(voxel_feat, voxel_feat_1, spatial_loc, training=False,
                                        src_mask=voxel_grid_out[-1], tgt_mask=voxel_grid_1[-1])

            # torch.save(voxel_grid_1[-1].cpu(), os.path.join(savedir, f'vox_grid_seq_{offset_idx}.pt'))
            np.save(os.path.join(savedir, f'vox_grid_seq_{offset_idx}.npy'), voxel_grid_1[-1].cpu().numpy())
            voxel_grid_out = voxel_grid_1
            # torch.save((unit_flow * self.voxel_size).cpu(), os.path.join(savedir, f'vox_flow_seq_{idx}.pt'))
            np.save(os.path.join(savedir, f'vox_flow_seq_{idx}.npy'), (unit_flow * self.voxel_size).cpu().numpy())
            voxel_feat = voxel_feat_1
