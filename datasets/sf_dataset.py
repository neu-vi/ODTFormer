import os
import numpy as np
import torch
import warnings
from PIL import Image

from .data_io import get_transform, read_all_lines
from .IO import readPFM, readFlow
from .voxel_dataset import VoxelDataset
from models.wrappers import Camera, Pose


def arrays_not_none(*arrays):
    return all(_ is not None for _ in arrays)


class DrivingVoxelFLow(VoxelDataset):
    def __init__(self, datapath, list_filename, training, roi_scale, voxel_sizes, transform=True, *, filter_ground=True,
                 color_jitter=False, occupied_gates=(None, None, 10, 5), resize_shape=None):
        super().__init__(datapath, roi_scale, voxel_sizes, transform, filter_ground=filter_ground,
                         color_jitter=color_jitter, occupied_gates=occupied_gates, resize_shape=resize_shape)
        self.left_0_filenames = None
        self.left_1_filenames = None
        self.right_0_filenames = None
        self.right_1_filenames = None
        self.disp_0_filenames = None
        self.disp_1_filenames = None
        self.disp_change_filenames = None
        self.flow_filenames = None
        self.gt_voxel_0_filenames = None
        self.gt_voxel_1_filenames = None
        self.gt_flow_filenames = None
        self.focal_lengths = None
        self.calib_filepaths = None
        self.load_path(list_filename)
        if training:
            assert arrays_not_none(self.disp_0_filenames, self.disp_1_filenames, self.disp_change_filenames,
                                   self.flow_filenames)

        self.file_id = None
        self.baseline = 1.0
        self.ground_y = 0.9
        self.img_res = (960, 540)

    def load_path(self, list_filename):
        # Format - [left_0, right_0, left_1, right_1, disp_0, disp_1, disp_change, oflow], gt_voxel_0, gt_voxel_1,
        # gt_flow, [focal_length, calib]
        lines = read_all_lines(list_filename)
        splits = [line.split() for line in lines]
        self.left_0_filenames = []
        self.right_0_filenames = []
        self.left_1_filenames = []
        self.right_1_filenames = []
        self.disp_0_filenames = []
        self.disp_1_filenames = []
        self.disp_change_filenames = []

        self.flow_filenames = []

        self.focal_lengths = []
        self.calib_filepaths = []
        for x in splits:
            self.left_0_filenames.append(x[0])
            self.right_0_filenames.append(x[1])
            self.left_1_filenames.append(x[2])
            self.right_1_filenames.append(x[3])
            self.disp_0_filenames.append(x[4])
            self.disp_1_filenames.append(x[5])
            self.disp_change_filenames.append(x[6])
            self.flow_filenames.append(x[7])

            self.focal_lengths.append(x[-2])
            self.calib_filepaths.append(x[-1])

        # stored gt available
        if len(splits[0]) > 10:
            self.stored_gt = True
            self.gt_voxel_0_filenames = []
            self.gt_voxel_1_filenames = []
            self.gt_flow_filenames = []
            for x in splits:
                self.gt_voxel_0_filenames.append(x[8])
                self.gt_voxel_1_filenames.append(x[9])
                self.gt_flow_filenames.append(x[10])

    def load_image(self, filename, *, store_id=False):
        if store_id:
            self.file_id = int(filename.split('/')[-1].split('.')[0])
        return Image.open(filename).convert('RGB')

    def load_disp(self, filename):
        data, _ = readPFM(filename)

        return data

    @staticmethod
    def load_flow(filename):
        data = readFlow(filename)

        return data

    def load_calib(self, filepath):
        line_idx_0 = 4 * (self.file_id - 1) + 1
        extrinsic_file = os.path.join(filepath, 'camera_data.txt')

        with open(extrinsic_file, 'r') as f:
            lines = f.readlines()
            assert int(lines[line_idx_0 - 1].split()[-1]) == self.file_id
            line_1_0 = lines[line_idx_0]
            line_3_0 = lines[line_idx_0 + 1]
            line_1_1 = lines[line_idx_0 + 4]
            line_3_1 = lines[line_idx_0 + 5]

        split_1_0 = line_1_0.split()
        split_3_0 = line_3_0.split()
        split_1_1 = line_1_1.split()
        split_3_1 = line_3_1.split()
        assert split_1_0[0] == 'L' and split_3_0[0] == 'R'
        assert split_1_1[0] == 'L' and split_3_1[0] == 'R'
        extrinsic_1_0 = np.array(list(map(float, split_1_0[1:]))).reshape(4, 4)
        extrinsic_3_0 = np.array(list(map(float, split_3_0[1:]))).reshape(4, 4)
        extrinsic_1_1 = np.array(list(map(float, split_1_1[1:]))).reshape(4, 4)
        extrinsic_3_1 = np.array(list(map(float, split_3_1[1:]))).reshape(4, 4)

        extrinsic_norm_0 = np.linalg.inv(extrinsic_1_0)
        extrinsic_norm_1 = np.linalg.inv(extrinsic_1_1)

        extrinsic_01_0 = extrinsic_1_0 @ extrinsic_norm_0
        extrinsic_03_0 = extrinsic_3_0 @ extrinsic_norm_0
        extrinsic_01_1 = extrinsic_1_1 @ extrinsic_norm_1
        extrinsic_03_1 = extrinsic_3_1 @ extrinsic_norm_1

        T_world_cam_01_0 = np.concatenate(
            [extrinsic_01_0[:3, :3].flatten(), [0., 0., 0.]], axis=-1)
        T_world_cam_03_0 = np.concatenate(
            [extrinsic_03_0[:3, :3].flatten(), extrinsic_03_0[:3, 3]], axis=-1)
        T_world_cam_01_1 = np.concatenate(
            [extrinsic_01_1[:3, :3].flatten(), [0., 0., 0.]], axis=-1)
        T_world_cam_03_1 = np.concatenate(
            [extrinsic_03_1[:3, :3].flatten(), extrinsic_03_1[:3, 3]], axis=-1)

        cam_01_0 = np.array([self.f_u, self.f_v, self.c_u, self.c_v])
        cam_03_0 = np.array([self.f_u, self.f_v, self.c_u, self.c_v])
        cam_01_1 = np.array([self.f_u, self.f_v, self.c_u, self.c_v])
        cam_03_1 = np.array([self.f_u, self.f_v, self.c_u, self.c_v])

        T_world_cam_101_0 = T_world_cam_01_0.astype(np.float32)
        cam_101_0 = cam_01_0.astype(np.float32)
        T_world_cam_103_0 = T_world_cam_03_0.astype(np.float32)
        cam_103_0 = cam_03_0.astype(np.float32)
        T_world_cam_101_1 = T_world_cam_01_1.astype(np.float32)
        cam_101_1 = cam_01_1.astype(np.float32)
        T_world_cam_103_1 = T_world_cam_03_1.astype(np.float32)
        cam_103_1 = cam_03_1.astype(np.float32)

        self.lidar_extrinsic = (Pose(T_world_cam_101_0), Pose(T_world_cam_101_1))
        return T_world_cam_101_0, cam_101_0, T_world_cam_103_0, cam_103_0, T_world_cam_101_1, cam_101_1, \
            T_world_cam_103_1, cam_103_1

    def project_image_to_rect_(self, uv_depth):
        x = (uv_depth[:, 0] - self.c_u) * uv_depth[:, 2] / self.f_u
        y = (uv_depth[:, 1] - self.c_v) * uv_depth[:, 2] / self.f_v
        pts_3d_rect = np.zeros_like(uv_depth)
        pts_3d_rect[:, 0] = x
        pts_3d_rect[:, 1] = y
        pts_3d_rect[:, 2] = uv_depth[:, 2]
        return pts_3d_rect

    def project_image_to_velo_(self, uv_depth, frame_id):
        return self.lidar_extrinsic[frame_id].inverse().transform(
            self.project_image_to_rect_(uv_depth)).numpy()

    def calc_cloud_flow(self, disp, frame_id, disp_change=None, flow=None):
        depth_gt = self.f_u * self.baseline / disp
        mask = (depth_gt > 0).reshape(-1)

        rows, cols = depth_gt.shape
        x, y = np.meshgrid(np.arange(cols, dtype=np.float32), np.arange(rows, dtype=np.float32))

        points = np.stack([x, y, depth_gt], axis=-1).reshape(-1, 3)
        points = points[mask]
        cloud = self.project_image_to_velo_(points, frame_id)
        if disp_change is not None and flow is not None:
            assert flow.shape[:-1] == depth_gt.shape == disp_change.shape
            depth_gt_1 = self.f_u * self.baseline / (disp + disp_change)
            x_1 = x + flow[..., 0]
            y_1 = y + flow[..., 1]
            points_1 = np.stack([x_1, y_1, depth_gt_1], axis=-1).reshape(-1, 3)
            valid_mask = (0 <= points_1[:, 0]) & (points_1[:, 0] <= self.img_res[0]) & (0 <= points_1[:, 1]) & (
                    points_1[:, 1] <= self.img_res[1])[mask]
            points_1 = points_1[mask]

            cloud_1 = self.project_image_to_velo_(points_1, frame_id)
            flow = (cloud_1 - cloud) * valid_mask[:, None]

            return cloud, flow
        return cloud

    def __len__(self):
        return len(self.left_0_filenames)

    def __getitem__(self, index):
        left_img_0_ = self.load_image(os.path.join(self.datapath, self.left_0_filenames[index]), store_id=True)
        right_img_0_ = self.load_image(os.path.join(self.datapath, self.right_0_filenames[index]))
        left_img_1_ = self.load_image(os.path.join(self.datapath, self.left_1_filenames[index]))
        right_img_1_ = self.load_image(os.path.join(self.datapath, self.right_1_filenames[index]))

        self.c_u = 479.5
        self.c_v = 269.5
        if self.focal_lengths[index] == '15mm':
            self.f_u = 450.0
            self.f_v = 450.0
        elif self.focal_lengths[index] == '35mm':
            self.f_u = 1050.0
            self.f_v = 1050.0
        else:
            raise RuntimeError('Unrecognizable camera focal length')
        T_world_cam_101_0, cam_101_0, T_world_cam_103_0, cam_103_0, T_world_cam_101_1, cam_101_1, T_world_cam_103_1, \
            cam_103_1 = self.load_calib(os.path.join(self.datapath, self.calib_filepaths[index]))
        disp_gt_0 = self.load_disp(os.path.join(self.datapath, self.disp_0_filenames[index]))
        disp_gt_1 = self.load_disp(os.path.join(self.datapath, self.disp_1_filenames[index]))
        disp_change = self.load_disp(os.path.join(self.datapath, self.disp_change_filenames[index]))
        optical_flow_gt = self.load_flow(os.path.join(self.datapath, self.flow_filenames[index]))

        # numpy to tensor
        T_world_cam_101_0 = torch.from_numpy(T_world_cam_101_0)
        T_world_cam_103_0 = torch.from_numpy(T_world_cam_103_0)
        T_world_cam_101_1 = torch.from_numpy(T_world_cam_101_1)
        T_world_cam_103_1 = torch.from_numpy(T_world_cam_103_1)

        w, h = left_img_0_.size
        crop_w, crop_h = self.img_res

        if self.resize_shape is None:
            self.resize_shape = (crop_h, crop_w)
        scale = self.resize_shape[1] / crop_w, self.resize_shape[0] / crop_h
        processed = get_transform(self.color_jitter, self.resize_shape)
        left_top = [0, 0]

        if self.transform:
            if w < crop_w:
                w_pad = crop_w - w
                h_pad = max(crop_h - h, 0)
                left_img_0 = Image.new(left_img_0_.mode, (crop_w, crop_h), (0, 0, 0))
                left_img_0.paste(left_img_0_, (0, 0))
                right_img_0 = Image.new(right_img_0_.mode, (crop_w, crop_h), (0, 0, 0))
                right_img_0.paste(right_img_0_, (0, 0))
                left_img_1 = Image.new(left_img_1_.mode, (crop_w, crop_h), (0, 0, 0))
                left_img_1.paste(left_img_1_, (0, 0))
                right_img_1 = Image.new(right_img_1_.mode, (crop_w, crop_h), (0, 0, 0))
                right_img_1.paste(right_img_1_, (0, 0))
                disp_gt_0 = np.lib.pad(
                    disp_gt_0, ((0, 0), (h_pad, w_pad)), mode='constant', constant_values=0)
                disp_gt_1 = np.lib.pad(
                    disp_gt_1, ((0, 0), (h_pad, w_pad)), mode='constant', constant_values=0)
                disp_change = np.lib.pad(
                    disp_change, ((0, 0), (h_pad, w_pad)), mode='constant', constant_values=0)
                optical_flow_gt = np.lib.pad(
                    optical_flow_gt, ((0, 0), (h_pad, w_pad), (0, 0)), mode='constant', constant_values=0)
            else:
                w_crop = w - crop_w
                h_crop = h - crop_h
                left_img_0 = left_img_0_.crop((w_crop, h_crop, w, h))
                right_img_0 = right_img_0_.crop((w_crop, h_crop, w, h))
                left_img_1 = left_img_1_.crop((w_crop, h_crop, w, h))
                right_img_1 = right_img_1_.crop((w_crop, h_crop, w, h))
                disp_gt_0 = disp_gt_0[h_crop: h, w_crop: w]
                disp_gt_1 = disp_gt_1[h_crop: h, w_crop: w]
                disp_change = disp_change[h_crop: h, w_crop: w]
                optical_flow_gt = optical_flow_gt[h_crop: h, w_crop: w]

                left_top = [w_crop, h_crop]

            left_img_0 = processed(left_img_0)
            right_img_0 = processed(right_img_0)
            left_img_1 = processed(left_img_1)
            right_img_1 = processed(right_img_1)
        else:
            w_crop = w - crop_w
            h_crop = h - crop_h
            left_img_0 = left_img_0_.crop((w_crop, h_crop, w, h))
            right_img_0 = right_img_0_.crop((w_crop, h_crop, w, h))
            left_img_1 = left_img_1_.crop((w_crop, h_crop, w, h))
            right_img_1 = right_img_1_.crop((w_crop, h_crop, w, h))
            left_img_0 = np.asarray(left_img_0)
            right_img_0 = np.asarray(right_img_0)
            left_img_1 = np.asarray(left_img_1)
            right_img_1 = np.asarray(right_img_1)
            left_top = [w_crop, h_crop]

        all_vox_grid_0_gt = []
        all_vox_grid_1_gt = []
        cloud_gt_0, flow_gt = self.calc_cloud_flow(disp_gt_0, 0, disp_change, optical_flow_gt)
        cloud_gt_1 = self.calc_cloud_flow(disp_gt_1, 1)
        cloud_gt_0 = np.concatenate([cloud_gt_0, flow_gt], axis=-1)

        filtered_cloud_gt_0 = self.filter_cloud(cloud_gt_0)
        filtered_cloud_gt_1 = self.filter_cloud(cloud_gt_1)

        if self.stored_gt:
            all_vox_grid_0_gt = self.load_gt(os.path.join(self.datapath, self.gt_voxel_0_filenames[index]))
            all_vox_grid_1_gt = self.load_gt(os.path.join(self.datapath, self.gt_voxel_1_filenames[index]))
            vox_flow_gt = self.load_gt(os.path.join(self.datapath, self.gt_flow_filenames[index]))
            valid_gt_0, _ = self.calc_voxel_grid(filtered_cloud_gt_0, 0, occupied_gate=self.occupied_gates[0])
            # if not torch.allclose(all_vox_grid_0_gt[0], torch.from_numpy(valid_gt_0)):
            #     warnings.warn(
            #         f'Stored label inconsistent.\n Loaded gt: \n {all_vox_grid_0_gt[0]} \n Validate gt: \n '
            #         f'{valid_gt_0}')
        else:
            parent_grid_0 = None
            parent_grid_1 = None
            vox_flow_gt = None
            try:
                for level in range(len(self.grid_sizes)):
                    occupied_gate = self.occupied_gates[level]
                    if level == len(self.grid_sizes) - 1:
                        vox_grid_gt_0, cloud_np_gt_0, vox_flow_gt = self.calc_voxel_grid(
                            filtered_cloud_gt_0, level=level, parent_grid=parent_grid_0,
                            occupied_gate=occupied_gate, get_flow=True)
                    else:
                        vox_grid_gt_0, cloud_np_gt_0 = self.calc_voxel_grid(
                            filtered_cloud_gt_0, level=level, parent_grid=parent_grid_0,
                            occupied_gate=occupied_gate)

                    vox_grid_gt_1, cloud_np_gt_1 = self.calc_voxel_grid(
                        filtered_cloud_gt_1, level=level, parent_grid=parent_grid_1, occupied_gate=occupied_gate)
                    vox_grid_gt_0 = torch.from_numpy(vox_grid_gt_0)
                    vox_grid_gt_1 = torch.from_numpy(vox_grid_gt_1)

                    parent_grid_0 = vox_grid_gt_0
                    parent_grid_1 = vox_grid_gt_1
                    all_vox_grid_0_gt.append(vox_grid_gt_0)
                    all_vox_grid_1_gt.append(vox_grid_gt_1)
            except Exception as e:
                raise RuntimeError('Error in calculating voxel grids from point cloud')

        imc, imh, imw = left_img_0.shape
        cam_101_0 = Camera(np.concatenate(([imw, imh], cam_101_0)).astype(np.float32))
        cam_101_0 = cam_101_0.crop(left_top, torch.tensor([crop_w, crop_h]))
        cam_101_0 = cam_101_0.scale(scale)
        cam_103_0 = Camera(np.concatenate(([imw, imh], cam_103_0)).astype(np.float32))
        cam_103_0 = cam_103_0.crop(left_top, torch.tensor([crop_w, crop_h]))
        cam_103_0 = cam_103_0.scale(scale)
        cam_101_1 = Camera(np.concatenate(([imw, imh], cam_101_1)).astype(np.float32))
        cam_101_1 = cam_101_1.crop(left_top, torch.tensor([crop_w, crop_h]))
        cam_101_1 = cam_101_1.scale(scale)
        cam_103_1 = Camera(np.concatenate(([imw, imh], cam_103_1)).astype(np.float32))
        cam_103_1 = cam_103_1.crop(left_top, torch.tensor([crop_w, crop_h]))
        cam_103_1 = cam_103_1.scale(scale)

        return {'left_0': left_img_0,
                'right_0': right_img_0,
                'left_1': left_img_1,
                'right_1': right_img_1,
                'T_world_cam_101_0': T_world_cam_101_0,
                'cam_101_0': cam_101_0.data,
                'T_world_cam_103_0': T_world_cam_103_0,
                'cam_103_0': cam_103_0.data,
                'T_world_cam_101_1': T_world_cam_101_1,
                'cam_101_1': cam_101_1.data,
                'T_world_cam_103_1': T_world_cam_103_1,
                'cam_103_1': cam_103_1.data,
                'voxel_grid_0': all_vox_grid_0_gt,
                'voxel_grid_1': all_vox_grid_1_gt,
                'voxel_flow': vox_flow_gt,
                'point_cloud_0': filtered_cloud_gt_0.astype(np.float32).tobytes(),
                'point_cloud_1': filtered_cloud_gt_1.astype(np.float32).tobytes(),
                "left_filename": self.left_0_filenames[index]}
