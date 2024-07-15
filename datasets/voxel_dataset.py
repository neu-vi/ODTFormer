import numpy as np
from PIL import Image
import cv2
import torch
from torch.utils.data import Dataset

from models import ref_points_generator


class VoxelDataset(Dataset):
    def __init__(self, datapath, roi_scale, voxel_sizes, transform, *, filter_ground, color_jitter, occupied_gates,
                 resize_shape):
        self.datapath = datapath
        self.stored_gt = False
        # initialize as null
        self.c_u = None
        self.c_v = None
        self.f_u = None
        self.f_v = None
        self.lidar_extrinsic = None
        self.roi_scale = roi_scale  # [min_x, max_x, min_y, max_y, min_z, max_z]
        assert len(voxel_sizes) == 4, 'Incomplete voxel sizes for 4 levels.'
        self.voxel_sizes = voxel_sizes

        self.grid_sizes = []
        for voxel_size in self.voxel_sizes:
            range_x = self.roi_scale[1] - self.roi_scale[0]
            range_y = self.roi_scale[3] - self.roi_scale[2]
            range_z = self.roi_scale[5] - self.roi_scale[4]
            if range_x % voxel_size != 0 or range_y % voxel_size != 0 or range_z % voxel_size != 0:
                raise RuntimeError('Voxel volume range indivisible by voxel sizes.')

            grid_size_x = int(range_x // voxel_size)
            grid_size_y = int(range_y // voxel_size)
            grid_size_z = int(range_z // voxel_size)
            self.grid_sizes.append((grid_size_x, grid_size_y, grid_size_z))

        self.transform = transform
        # if ground y > ground_y will be filtered
        self.filter_ground = filter_ground
        self.ground_y = None
        self.color_jitter = color_jitter
        self.occupied_gates = occupied_gates
        self.resize_shape = resize_shape

    def load_path(self, list_filename):
        raise NotImplementedError

    @staticmethod
    def load_image(filename):
        return Image.open(filename).convert('RGB')

    @staticmethod
    def load_disp(filename):
        # 16 bit Grayscale
        data = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
        out = data.astype(np.float32) / 256.
        return out

    load_depth = load_disp

    @staticmethod
    def load_flow(filename):
        raise NotImplementedError

    @staticmethod
    def load_gt(filename):
        return torch.load(filename)

    def load_calib(self, filename):
        raise NotImplementedError

    def project_image_to_rect(self, uv_depth):
        x = (uv_depth[:, 0] - self.c_u) * uv_depth[:, 2] / self.f_u
        y = (uv_depth[:, 1] - self.c_v) * uv_depth[:, 2] / self.f_v
        pts_3d_rect = np.zeros_like(uv_depth)
        pts_3d_rect[:, 0] = x
        pts_3d_rect[:, 1] = y
        pts_3d_rect[:, 2] = uv_depth[:, 2]
        return pts_3d_rect

    def project_image_to_velo(self, uv_depth):
        return self.lidar_extrinsic.inverse().transform(self.project_image_to_rect(uv_depth)).numpy()

    def filter_cloud(self, cloud):
        min_mask = cloud[..., :3] >= [self.roi_scale[0], self.roi_scale[2], self.roi_scale[4]]
        if self.filter_ground and self.roi_scale[3] > self.ground_y:
            max_mask = cloud[..., :3] <= [self.roi_scale[1], self.ground_y, self.roi_scale[5]]
        else:
            max_mask = cloud[..., :3] <= [self.roi_scale[1], self.roi_scale[3], self.roi_scale[5]]
        min_mask = min_mask[:, 0] & min_mask[:, 1] & min_mask[:, 2]
        max_mask = max_mask[:, 0] & max_mask[:, 1] & max_mask[:, 2]
        filter_mask = min_mask & max_mask
        filtered_cloud = cloud[filter_mask]
        return filtered_cloud

    def calc_voxel_grid(self, filtered_cloud, level, parent_grid=None, get_flow=False,
                        *,
                        rtol: float = 0.3):
        occupied_gate_ = self.occupied_gates[level]
        occupied_gate = occupied_gate_ if occupied_gate_ is not None else 1
        assert occupied_gate > 0

        vox_size = self.voxel_sizes[level]
        reference_points = ref_points_generator([self.roi_scale[0], self.roi_scale[2], self.roi_scale[4]],
                                                self.grid_sizes[level], vox_size, normalize=False).view(-1, 3).numpy()

        if parent_grid is not None:
            search_mask = parent_grid[:, None, :, None, :, None].repeat(1, 2, 1, 2, 1, 2).view(-1).to(
                bool).numpy()
        else:
            search_mask = torch.ones(reference_points.shape[0]).to(bool)

        # num_search_grids, num_pc - bool
        vox_hits = np.bitwise_and.reduce(
            np.abs(filtered_cloud[..., None, :3] - reference_points[search_mask]) <= vox_size / 2,
            axis=-1)
        # num_search_grids - bool
        valid_hits = np.sum(vox_hits, axis=0) >= occupied_gate
        occupied_grid = np.zeros(reference_points.shape[0])
        occupied_grid[search_mask] = valid_hits.astype(int)

        if not get_flow:
            return occupied_grid.reshape(*self.grid_sizes[level]), reference_points[occupied_grid.astype(bool)]
        else:
            assert filtered_cloud.shape[-1] == 6
            mean_flow = vox_hits.T @ filtered_cloud[..., 3:] / (np.sum(vox_hits, axis=0, keepdims=True).T + 1e-5)
            mean_flow = np.round(mean_flow, decimals=1)
            sflow = np.zeros(reference_points.shape)
            sflow[search_mask] = (mean_flow - rtol * np.sign(mean_flow) * vox_size) // vox_size * vox_size
            sflow *= occupied_grid[..., None]

            return occupied_grid.reshape(*self.grid_sizes[level]), reference_points[
                occupied_grid.astype(bool)], sflow.reshape(*self.grid_sizes[level], 3)
