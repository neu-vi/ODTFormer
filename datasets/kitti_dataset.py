import os
import numpy as np
import torch
import warnings
from PIL import Image

from .data_io import get_transform, read_all_lines
from .voxel_dataset import VoxelDataset
from models.wrappers import Camera, Pose


class VoxelKITTIDataset(VoxelDataset):
    def __init__(self, datapath, list_filename, training, roi_scale, voxel_sizes, transform=True, *,
                 filter_ground=True, color_jitter=False, occupied_gates=(20, 20, 10, 5), resize_shape=None):
        super().__init__(datapath, roi_scale, voxel_sizes, transform, filter_ground=filter_ground,
                         color_jitter=color_jitter, occupied_gates=occupied_gates, resize_shape=resize_shape)
        self.left_filenames = None
        self.right_filenames = None
        self.disp_filenames = None
        self.gt_voxel_filenames = None
        self.calib_filenames = None
        self.load_path(list_filename)
        if training:
            assert self.disp_filenames is not None

        # Camera intrinsics
        self.baseline = 0.54
        self.ground_y = 1.5

    def load_path(self, list_filename):
        lines = read_all_lines(list_filename)
        splits = [line.split() for line in lines]
        self.left_filenames = []
        self.right_filenames = []
        self.calib_filenames = []
        for x in splits:
            self.left_filenames.append(x[0])
            self.right_filenames.append(x[1])
            self.calib_filenames.append(x[-1])

        # with gt disp and flow
        if len(splits[0]) >= 4:
            self.disp_filenames = [x[2] for x in splits]

            # stored gt available
            if len(splits[0]) > 4:
                self.stored_gt = True
                self.gt_voxel_filenames = [x[3] for x in splits]

    def load_calib(self, filename):
        with open(filename, 'r') as f:
            lines = f.readlines()

        R_02 = None
        T_02 = None
        P_rect_02 = None
        R_rect_02 = None
        R_03 = None
        T_03 = None
        P_rect_03 = None
        R_rect_03 = None
        for line in lines:
            splits = line.split()
            if splits[0] == 'R_00:':
                R_02 = np.array(list(map(float, splits[1:]))).reshape(3, 3)
            elif splits[0] == 'T_00:':
                T_02 = np.array(list(map(float, splits[1:])))
            elif splits[0] == 'P_rect_00:':
                P_rect_02 = np.array(list(map(float, splits[1:]))).reshape(3, 4)
            elif splits[0] == 'R_rect_00:':
                R_rect_02 = np.array(list(map(float, splits[1:]))).reshape(3, 3)
            elif splits[0] == 'R_01:':
                R_03 = np.array(list(map(float, splits[1:]))).reshape(3, 3)
            elif splits[0] == 'T_01:':
                T_03 = np.array(list(map(float, splits[1:])))
            elif splits[0] == 'P_rect_03:':
                P_rect_03 = np.array(list(map(float, splits[1:]))).reshape(3, 4)
            elif splits[0] == 'R_rect_03:':
                R_rect_03 = np.array(list(map(float, splits[1:]))).reshape(3, 3)

        # 4x4
        Rt_02 = np.concatenate([R_02, np.expand_dims(T_02, axis=-1)], axis=-1)
        Rt_02 = np.concatenate([Rt_02, np.array([[0., 0., 0., 1.]])], axis=0)
        Rt_03 = np.concatenate([R_03, np.expand_dims(T_03, axis=-1)], axis=-1)
        Rt_03 = np.concatenate([Rt_03, np.array([[0., 0., 0., 1.]])], axis=0)

        R_rect_02 = np.concatenate([R_rect_02, np.array([[0., 0., 0.]]).T], axis=-1)
        R_rect_02 = np.concatenate([R_rect_02, np.array([[0., 0., 0., 1.]])], axis=0)
        R_rect_03 = np.concatenate([R_rect_03, np.array([[0., 0., 0.]]).T], axis=-1)
        R_rect_03 = np.concatenate([R_rect_03, np.array([[0., 0., 0., 1.]])], axis=0)

        T_world_cam_02 = R_rect_02 @ Rt_02
        T_world_cam_02 = np.concatenate([T_world_cam_02[:3, :3].flatten(), T_world_cam_02[:3, 3]], axis=-1)
        T_world_cam_03 = R_rect_03 @ Rt_03
        T_world_cam_03 = np.concatenate([T_world_cam_03[:3, :3].flatten(), T_world_cam_03[:3, 3]], axis=-1)

        self.c_u = P_rect_02[0, 2]
        self.c_v = P_rect_02[1, 2]
        self.f_u = P_rect_02[0, 0]
        self.f_v = P_rect_02[1, 1]

        cam_02 = np.array([P_rect_02[0, 0], P_rect_02[1, 1], P_rect_02[0, 2], P_rect_02[1, 2]])
        cam_03 = np.array([P_rect_03[0, 0], P_rect_03[1, 1], P_rect_03[0, 2], P_rect_03[1, 2]])

        T_world_cam_101 = T_world_cam_02.astype(np.float32)
        cam_101 = cam_02.astype(np.float32)
        T_world_cam_103 = T_world_cam_03.astype(np.float32)
        cam_103 = cam_03.astype(np.float32)

        self.lidar_extrinsic = Pose(T_world_cam_101)

        return T_world_cam_101, cam_101, T_world_cam_103, cam_103

    def calc_cloud(self, disparity):
        depth_gt = self.f_u * self.baseline / (disparity + 1e-5)
        mask = (disparity > 0).reshape(-1)

        rows, cols = depth_gt.shape
        x, y = np.meshgrid(np.arange(cols, dtype=np.float32), np.arange(rows, dtype=np.float32))

        points = np.stack([x, y, depth_gt], axis=-1).reshape(-1, 3)
        points = points[mask]

        cloud = self.project_image_to_velo(points)
        return cloud

    def __len__(self):
        return len(self.left_filenames)

    def __getitem__(self, index):
        left_img_ = self.load_image(os.path.join(self.datapath, self.left_filenames[index]))
        right_img_ = self.load_image(os.path.join(self.datapath, self.right_filenames[index]))
        T_world_cam_101, cam_101, T_world_cam_103, cam_103 = self.load_calib(
            os.path.join(self.datapath, self.calib_filenames[index]))
        disp_gt = None
        if self.disp_filenames is not None:
            disp_gt = self.load_disp(os.path.join(self.datapath, self.disp_filenames[index]))

        # numpy to tensor
        T_world_cam_101 = torch.from_numpy(T_world_cam_101)
        T_world_cam_103 = torch.from_numpy(T_world_cam_103)

        w, h = left_img_.size
        crop_w, crop_h = 1224, 370

        if self.resize_shape is None:
            self.resize_shape = (crop_h, crop_w)
        scale = self.resize_shape[1] / crop_w, self.resize_shape[0] / crop_h
        processed = get_transform(self.color_jitter, self.resize_shape)
        left_top = [0, 0]

        if self.transform:
            if w < crop_w:
                w_pad = crop_w - w
                h_pad = max(crop_h - h, 0)
                left_img = Image.new(left_img_.mode, (crop_w, crop_h), (0, 0, 0))
                left_img.paste(left_img_, (0, 0))
                right_img = Image.new(right_img_.mode, (crop_w, crop_h), (0, 0, 0))
                right_img.paste(right_img_, (0, 0))
                disp_gt = np.lib.pad(
                    disp_gt, ((0, 0), (h_pad, w_pad)), mode='constant', constant_values=0)

                left_img = processed(left_img)
                right_img = processed(right_img)
            else:
                w_crop = w - crop_w
                h_crop = h - crop_h
                left_img = left_img_.crop((w_crop, h_crop, w, h))
                right_img = right_img_.crop((w_crop, h_crop, w, h))
                if disp_gt is not None:
                    disp_gt = disp_gt[h_crop: h, w_crop: w]

                left_img = processed(left_img)
                right_img = processed(right_img)
                left_top = [w_crop, h_crop]
        else:
            w_crop = w - crop_w
            h_crop = h - crop_h
            left_img = left_img_.crop((w_crop, h_crop, w, h))
            right_img = right_img_.crop((w_crop, h_crop, w, h))
            left_img = np.asarray(left_img)
            right_img = np.asarray(right_img)
            left_top = [w_crop, h_crop]

        filtered_cloud_gt = None
        all_vox_grid_gt = []
        if disp_gt is not None:
            cloud_gt = self.calc_cloud(disp_gt)
            filtered_cloud_gt = self.filter_cloud(cloud_gt)

            if self.stored_gt:
                all_vox_grid_gt = self.load_gt(os.path.join(self.datapath, self.gt_voxel_filenames[index]))
                # ===== Different occlusion handling technique when generating gt labels =====
                # valid_gt, _ = self.calc_voxel_grid(filtered_cloud_gt, 0)
                # if not torch.allclose(all_vox_grid_gt[0], torch.from_numpy(valid_gt)):
                #     warnings.warn(
                #         f'Stored label inconsistent.\n Loaded gt: \n {all_vox_grid_gt[0]} \n Validate gt: \n'
                #         f'{valid_gt}')
            else:
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

        cam_101 = Camera(torch.tensor(np.concatenate(([w, h], cam_101)).astype(np.float32)))
        cam_101 = cam_101.crop(left_top, torch.tensor([crop_w, crop_h]))
        cam_101 = cam_101.scale(scale)
        cam_103 = Camera(torch.tensor(np.concatenate(([w, h], cam_103)).astype(np.float32)))
        cam_103 = cam_103.crop(left_top, torch.tensor([crop_w, crop_h]))
        cam_103 = cam_103.scale(scale)

        return {'left': left_img,
                'right': right_img,
                'T_world_cam_101': T_world_cam_101,
                'cam_101': cam_101.data,
                'T_world_cam_103': T_world_cam_103,
                'cam_103': cam_103.data,
                'voxel_grid': all_vox_grid_gt if len(all_vox_grid_gt) >= 0 else 'null',
                'point_cloud': filtered_cloud_gt.astype(
                    np.float32).tobytes() if filtered_cloud_gt is not None else 'null',
                "left_filename": self.left_filenames[index]}


class VoxelKITTIRaw(VoxelKITTIDataset):
    def __init__(self, datapath, list_filename, training, roi_scale, voxel_sizes, transform=True, *,
                 filter_ground=True, color_jitter=False, occupied_gates=(20, 20, 10, 5), resize_shape=None):
        self.velo2cam_filenames = None
        self.velo_filenames = None
        self.velo2cam = None
        super().__init__(datapath, list_filename, training, roi_scale, voxel_sizes, transform,
                         filter_ground=filter_ground, color_jitter=color_jitter, occupied_gates=occupied_gates,
                         resize_shape=resize_shape)

    def load_path(self, list_filename):
        lines = read_all_lines(list_filename)
        splits = [line.split() for line in lines]
        # left, right, scan, gt, velo2cam, calib
        self.left_filenames = []
        self.right_filenames = []
        self.velo2cam_filenames = []
        self.calib_filenames = []
        for x in splits:
            self.left_filenames.append(x[0])
            self.right_filenames.append(x[1])
            self.velo2cam_filenames.append(x[-2])
            self.calib_filenames.append(x[-1])

        # with gt velo
        if len(splits[0]) >= 5:
            self.velo_filenames = [x[2] for x in splits]
            # parent compatible
            self.disp_filenames = 1

            # stored gt available
            if len(splits[0]) > 5:
                self.stored_gt = True
                self.gt_voxel_filenames = [x[3] for x in splits]

    def load_velo2cam(self, filename):
        with open(filename, 'r') as f:
            lines = f.readlines()

        R_mat = None
        T_mat = None
        for line in lines:
            splits = line.split()
            if splits[0] == 'R:':
                R_mat = np.array(list(map(float, splits[1:])))
            elif splits[0] == 'T:':
                T_mat = np.array(list(map(float, splits[1:])))

        self.velo2cam = Pose(np.concatenate([R_mat, T_mat], axis=-1).astype(np.float32))

    def load_velo(self, filename):
        scan = np.fromfile(filename, dtype=np.float32).reshape((-1, 4))[:, :3]
        scan = self.velo2cam.transform(torch.from_numpy(scan))

        return scan

    def __getitem__(self, index):
        left_img_ = self.load_image(os.path.join(self.datapath, self.left_filenames[index]))
        right_img_ = self.load_image(os.path.join(self.datapath, self.right_filenames[index]))
        T_world_cam_101, cam_101, T_world_cam_103, cam_103 = self.load_calib(
            os.path.join(self.datapath, self.calib_filenames[index]))
        self.load_velo2cam(os.path.join(self.datapath, self.velo2cam_filenames[index]))
        scan_gt = None
        if self.velo_filenames is not None:
            scan_gt = self.load_velo(os.path.join(self.datapath, self.velo_filenames[index]))

        # numpy to tensor
        T_world_cam_101 = torch.from_numpy(T_world_cam_101)
        T_world_cam_103 = torch.from_numpy(T_world_cam_103)

        w, h = left_img_.size
        crop_w, crop_h = 1224, 370

        if self.resize_shape is None:
            self.resize_shape = (crop_h, crop_w)
        scale = self.resize_shape[1] / crop_w, self.resize_shape[0] / crop_h
        processed = get_transform(self.color_jitter, self.resize_shape)
        left_top = [0, 0]

        if self.transform:
            if w < crop_w:
                left_img = Image.new(left_img_.mode, (crop_w, crop_h), (0, 0, 0))
                left_img.paste(left_img_, (0, 0))
                right_img = Image.new(right_img_.mode, (crop_w, crop_h), (0, 0, 0))
                right_img.paste(right_img_, (0, 0))

                left_img = processed(left_img)
                right_img = processed(right_img)
            else:
                w_crop = w - crop_w
                h_crop = h - crop_h
                left_img = left_img_.crop((w_crop, h_crop, w, h))
                right_img = right_img_.crop((w_crop, h_crop, w, h))

                left_img = processed(left_img)
                right_img = processed(right_img)
                left_top = [w_crop, h_crop]
        else:
            w_crop = w - crop_w
            h_crop = h - crop_h
            left_img = left_img_.crop((w_crop, h_crop, w, h))
            right_img = right_img_.crop((w_crop, h_crop, w, h))
            left_img = np.asarray(left_img)
            right_img = np.asarray(right_img)
            left_top = [w_crop, h_crop]

        cam_101 = Camera(torch.tensor(np.concatenate(([w, h], cam_101)).astype(np.float32)))
        cam_101 = cam_101.crop(left_top, [crop_w, crop_h])
        cam_101 = cam_101.scale(scale)
        cam_103 = Camera(torch.tensor(np.concatenate(([w, h], cam_103)).astype(np.float32)))
        cam_103 = cam_103.crop(left_top, [crop_w, crop_h])
        cam_103 = cam_103.scale(scale)

        filtered_cloud_gt = None
        all_vox_grid_gt = []
        if scan_gt is not None:
            scan_cam_101 = self.lidar_extrinsic.transform(scan_gt).numpy()
            valid_scan = cam_101.project(scan_cam_101)[1]
            cloud_gt = scan_cam_101[valid_scan]
            filtered_cloud_gt = self.filter_cloud(cloud_gt)

            if self.stored_gt:
                all_vox_grid_gt = self.load_gt(os.path.join(self.datapath, self.gt_voxel_filenames[index]))
                valid_gt, _ = self.calc_voxel_grid(filtered_cloud_gt, 0)
                if not torch.allclose(all_vox_grid_gt[0], torch.from_numpy(valid_gt)):
                    warnings.warn(
                        f'Stored label inconsistent.\n Loaded gt: \n {all_vox_grid_gt[0]} \n Validate gt: \n'
                        f'{valid_gt}')
            else:
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

        return {'left': left_img,
                'right': right_img,
                'T_world_cam_101': T_world_cam_101,
                'cam_101': cam_101.data,
                'T_world_cam_103': T_world_cam_103,
                'cam_103': cam_103.data,
                'voxel_grid': all_vox_grid_gt if len(all_vox_grid_gt) >= 0 else 'null',
                'point_cloud': filtered_cloud_gt.astype(
                    np.float32).tobytes() if filtered_cloud_gt is not None else 'null',
                "left_filename": self.left_filenames[index]}
