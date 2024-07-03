import os
import numpy as np
import torch
import warnings

from .data_io import get_transform, read_all_lines
from .voxel_dataset import VoxelDataset
from models.wrappers import Camera, Pose


class VoxelDSDatasetCalib(VoxelDataset):
    def __init__(self, datapath, list_filename, training, roi_scale, voxel_sizes, transform=True, *,
                 filter_ground=True, color_jitter=False, occupied_gates=(20, 20, 20, 10)):
        super().__init__(datapath, roi_scale, voxel_sizes, transform, filter_ground=filter_ground,
                         color_jitter=color_jitter, occupied_gates=occupied_gates)
        self.left_filenames, self.right_filenames, self.depth_filenames, self.gt_filenames, self.calib_filenames = \
            self.load_path(list_filename)
        if training:
            assert self.depth_filenames is not None

        self.ground_y = 1

    def load_path(self, list_filename):
        lines = read_all_lines(list_filename)
        splits = [line.split() for line in lines]
        left_images = []
        right_images = []
        calib = []
        for x in splits:
            left_images.append(x[0])
            right_images.append(x[1])
            calib.append(x[-1])
        if len(splits[0]) == 3:  # ground truth not available
            return left_images, right_images, None, None, calib
        elif len(splits[0]) == 4:
            depth_map = [x[2] for x in splits]
            return left_images, right_images, depth_map, None, calib
        elif len(splits[0]) == 5:
            self.stored_gt = True
            depth_map = []
            gt_label = []
            for x in splits:
                depth_map.append(x[2])
                gt_label.append(x[3])
            return left_images, right_images, depth_map, gt_label, calib
        else:
            raise RuntimeError('Dataset filename format not supported.')

    def load_calib(self, filename):
        with open(filename, 'r') as f:
            lines = f.readlines()

        R_101 = None
        T_101 = None
        P_rect_101 = None
        R_rect_101 = None
        R_103 = None
        T_103 = None
        P_rect_103 = None
        R_rect_103 = None
        for line in lines:
            splits = line.split()
            if splits[0] == 'R_101:':
                R_101 = np.array(list(map(float, splits[1:]))).reshape(3, 3)
            elif splits[0] == 'T_101:':
                T_101 = np.array(list(map(float, splits[1:])))
            elif splits[0] == 'P_rect_101:':
                P_rect_101 = np.array(list(map(float, splits[1:]))).reshape(3, 4)
            elif splits[0] == 'R_rect_101:':
                R_rect_101 = np.array(list(map(float, splits[1:]))).reshape(3, 3)
            elif splits[0] == 'R_103:':
                R_103 = np.array(list(map(float, splits[1:]))).reshape(3, 3)
            elif splits[0] == 'T_103:':
                T_103 = np.array(list(map(float, splits[1:])))
            elif splits[0] == 'P_rect_103:':
                P_rect_103 = np.array(list(map(float, splits[1:]))).reshape(3, 4)
            elif splits[0] == 'R_rect_103:':
                R_rect_103 = np.array(list(map(float, splits[1:]))).reshape(3, 3)

        # 4x4
        Rt_101 = np.concatenate([R_101, np.expand_dims(T_101, axis=-1)], axis=-1)
        Rt_101 = np.concatenate([Rt_101, np.array([[0., 0., 0., 1.]])], axis=0)
        Rt_103 = np.concatenate([R_103, np.expand_dims(T_103, axis=-1)], axis=-1)
        Rt_103 = np.concatenate([Rt_103, np.array([[0., 0., 0., 1.]])], axis=0)

        R_rect_101 = np.concatenate([R_rect_101, np.array([[0., 0., 0.]]).T], axis=-1)
        R_rect_101 = np.concatenate([R_rect_101, np.array([[0., 0., 0., 1.]])], axis=0)
        R_rect_103 = np.concatenate([R_rect_103, np.array([[0., 0., 0.]]).T], axis=-1)
        R_rect_103 = np.concatenate([R_rect_103, np.array([[0., 0., 0., 1.]])], axis=0)

        # T_world_cam_101 = P_rect_101 @ R_rect_101 @ Rt_101
        T_world_cam_101 = R_rect_101 @ Rt_101
        T_world_cam_101 = np.concatenate([T_world_cam_101[:3, :3].flatten(), T_world_cam_101[:3, 3]], axis=-1)
        # T_world_cam_103 = P_rect_103 @ R_rect_103 @ Rt_103
        T_world_cam_103 = R_rect_103 @ Rt_103
        T_world_cam_103 = np.concatenate([T_world_cam_103[:3, :3].flatten(), T_world_cam_103[:3, 3]], axis=-1)

        self.c_u = P_rect_101[0, 2]
        self.c_v = P_rect_101[1, 2]
        self.f_u = P_rect_101[0, 0]
        self.f_v = P_rect_101[1, 1]

        cam_101 = np.array([P_rect_101[0, 0], P_rect_101[1, 1], P_rect_101[0, 2], P_rect_101[1, 2]])
        cam_103 = np.array([P_rect_103[0, 0], P_rect_103[1, 1], P_rect_103[0, 2], P_rect_103[1, 2]])

        T_world_cam_101 = T_world_cam_101.astype(np.float32)
        cam_101 = cam_101.astype(np.float32)
        T_world_cam_103 = T_world_cam_103.astype(np.float32)
        cam_103 = cam_103.astype(np.float32)

        self.lidar_extrinsic = Pose(T_world_cam_101)

        return T_world_cam_101, cam_101, T_world_cam_103, cam_103

    def calc_cloud(self, depth, left_img=None):
        mask = (depth > 0).reshape(-1)
        rows, cols = depth.shape
        c, r = np.meshgrid(np.arange(cols), np.arange(rows))
        points = np.stack([c, r, depth], axis=-1)
        points = points.reshape(-1, 3)
        points = points[mask]
        cloud = self.project_image_to_velo(points)
        if left_img is not None:
            left_img = left_img.reshape(-1, 3)
            return np.concatenate([cloud, left_img[mask]], axis=-1)

        return cloud

    def __len__(self):
        return len(self.left_filenames)

    def __getitem__(self, index):
        left_img_ = self.load_image(os.path.join(self.datapath, self.left_filenames[index]))
        right_img_ = self.load_image(os.path.join(self.datapath, self.right_filenames[index]))
        T_world_cam_101, cam_101, T_world_cam_103, cam_103 = self.load_calib(
            os.path.join(self.datapath, self.calib_filenames[index]))
        depth_gt = self.load_depth(os.path.join(self.datapath, self.depth_filenames[index]))

        # numpy to tensor
        T_world_cam_101 = torch.from_numpy(T_world_cam_101)
        T_world_cam_103 = torch.from_numpy(T_world_cam_103)

        w, h = left_img_.size
        crop_w, crop_h = 880, 400

        processed = get_transform(self.color_jitter)
        left_top = [0, 0]

        if self.transform:
            if w < crop_w:
                left_img = processed(left_img_).numpy()
                right_img = processed(right_img_).numpy()

                w_pad = crop_w - w
                left_img = np.lib.pad(
                    left_img, ((0, 0), (0, 0), (0, w_pad)), mode='constant', constant_values=0)
                right_img = np.lib.pad(
                    right_img, ((0, 0), (0, 0), (0, w_pad)), mode='constant', constant_values=0)
                depth_gt = np.lib.pad(
                    depth_gt, ((0, 0), (0, w_pad)), mode='constant', constant_values=0)

                left_img = torch.Tensor(left_img)
                right_img = torch.Tensor(right_img)
            else:
                w_crop = w - crop_w
                h_crop = h - crop_h
                left_img_ = left_img_.crop((w_crop, h_crop, w, h))
                right_img_ = right_img_.crop((w_crop, h_crop, w, h))
                depth_gt = depth_gt[h_crop: h, w_crop: w]

                left_img = processed(left_img_)
                right_img = processed(right_img_)
                left_top = [w_crop, h_crop]
        else:
            w_crop = w - crop_w
            h_crop = h - crop_h
            left_img_ = left_img_.crop((w_crop, h_crop, w, h))
            right_img_ = right_img_.crop((w_crop, h_crop, w, h))
            left_img = np.asarray(left_img_)
            right_img = np.asarray(right_img_)
            left_top = [w_crop, h_crop]

        left_top = np.repeat(np.array([left_top]), repeats=2, axis=0)

        # canvas = np.zeros((400, 880, 3), dtype=np.float32)
        # left_img_ = np.asarray(left_img_)
        # canvas[:left_img_.shape[0], :left_img_.shape[1], :] = left_img_
        colored_cloud_gt = self.calc_cloud(depth_gt)  # , left_img=canvas)
        filtered_cloud_gt = self.filter_cloud(colored_cloud_gt[..., :3])

        if self.stored_gt:
            all_vox_grid_gt = self.load_gt(os.path.join(self.datapath, self.gt_filenames[index]))
            valid_gt, _ = self.calc_voxel_grid(filtered_cloud_gt, 0)
            if not torch.allclose(all_vox_grid_gt[0], torch.from_numpy(valid_gt)):
                warnings.warn(
                    f'Stored label inconsistent.\n Loaded gt: \n {all_vox_grid_gt[0]} \n Validate gt: \n {valid_gt}')
        else:
            all_vox_grid_gt = []
            parent_grid = None
            try:
                for level in range(len(self.grid_sizes)):
                    vox_grid_gt, cloud_np_gt = self.calc_voxel_grid(
                        filtered_cloud_gt, level=level, parent_grid=parent_grid)
                    vox_grid_gt = torch.from_numpy(vox_grid_gt)

                    parent_grid = vox_grid_gt
                    all_vox_grid_gt.append(vox_grid_gt)
            except Exception as e:
                raise RuntimeError('Error in calculating voxel grids from point cloud')

        imc, imh, imw = left_img.shape
        cam_101 = np.concatenate(([imw, imh], cam_101)).astype(np.float32)
        cam_103 = np.concatenate(([imw, imh], cam_103)).astype(np.float32)

        return {"left": left_img,
                "right": right_img,
                'T_world_cam_101': T_world_cam_101,
                'cam_101': cam_101,
                'T_world_cam_103': T_world_cam_103,
                'cam_103': cam_103,
                # "depth": depth_gt,
                "voxel_grid": all_vox_grid_gt,
                'point_cloud': filtered_cloud_gt.astype(np.float32).tobytes(),
                # 'colored_point_cloud': colored_cloud_gt.astype(np.float32).tobytes(),
                'left_top': left_top,
                "left_filename": self.left_filenames[index]}
