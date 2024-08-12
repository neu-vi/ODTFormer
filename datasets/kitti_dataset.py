import os
import numpy as np
import torch
import warnings
from PIL import Image
import cv2

from .data_io import get_transform, read_all_lines
from .voxel_dataset import VoxelDataset
from models.wrappers import Camera, Pose


def arrays_not_none(*arrays):
    return all(_ is not None for _ in arrays)


class KITTIVoxelFLow(VoxelDataset):
    def __init__(self, datapath, list_filename, training, roi_scale, voxel_sizes, transform=True, *,
                 filter_ground=True, color_jitter=False, occupied_gates=(20, 20, 10, 5), resize_shape=None):
        super().__init__(datapath, roi_scale, voxel_sizes, transform, filter_ground=filter_ground,
                         color_jitter=color_jitter, occupied_gates=occupied_gates, resize_shape=resize_shape)
        self.left_0_filenames = None
        self.left_1_filenames = None
        self.right_0_filenames = None
        self.right_1_filenames = None
        self.disp_0_filenames = None
        self.disp_1_filenames = None
        self.flow_filenames = None
        self.gt_voxel_0_filenames = None
        self.gt_voxel_1_filenames = None
        self.gt_flow_filenames = None
        self.calib_filenames = None
        self.load_path(list_filename)
        if training:
            assert arrays_not_none(self.disp_0_filenames, self.disp_1_filenames, self.flow_filenames)

        # Camera intrinsics
        self.baseline = 0.54

        self.ground_y = 1.5

    def load_path(self, list_filename):
        # Format - left_0, right_0, left_1, right_1, disp_0, disp_1, flow, gt_voxel_0, gt_voxel_1, gt_flow, calib
        lines = read_all_lines(list_filename)
        splits = [line.split() for line in lines]
        self.left_0_filenames = []
        self.right_0_filenames = []
        self.left_1_filenames = []
        self.right_1_filenames = []
        self.calib_filenames = []
        for x in splits:
            self.left_0_filenames.append(x[0])
            self.right_0_filenames.append(x[1])
            self.left_1_filenames.append(x[2])
            self.right_1_filenames.append(x[3])
            self.calib_filenames.append(x[-1])

        # with gt disp and flow
        if len(splits[0]) >= 8:
            self.disp_0_filenames = []
            self.disp_1_filenames = []
            self.flow_filenames = []
            for x in splits:
                self.disp_0_filenames.append(x[4])
                self.disp_1_filenames.append(x[5])
                self.flow_filenames.append(x[6])

            # stored gt available
            if len(splits[0]) == 11:
                self.stored_gt = True
                self.gt_voxel_0_filenames = []
                self.gt_voxel_1_filenames = []
                self.gt_flow_filenames = []
                for x in splits:
                    self.gt_voxel_0_filenames.append(x[7])
                    self.gt_voxel_1_filenames.append(x[8])
                    self.gt_flow_filenames.append(x[9])

    @staticmethod
    def load_flow(filename):
        # 48 bit RGB
        data = cv2.imread(filename, cv2.IMREAD_UNCHANGED)[:, :, ::-1]
        out = np.zeros_like(data, dtype=np.float32)
        out[..., 0] = (data[..., 0].astype(np.float32) - 2 ** 15) / 64.
        out[..., 1] = (data[..., 1].astype(np.float32) - 2 ** 15) / 64.
        out[..., -1] = data[..., -1]

        return out

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

    def calc_cloud_flow(self, disparity_0, disparity_1, flow):
        depth_gt_0 = self.f_u * self.baseline / (disparity_0 + 1e-5)
        depth_gt_1 = self.f_u * self.baseline / (disparity_1 + 1e-5)
        mask_0 = disparity_0 > 0
        mask_1 = disparity_1 > 0
        occ_mask = mask_0 ^ mask_1

        rows, cols = depth_gt_0.shape
        x, y = np.meshgrid(np.arange(cols, dtype=np.float32), np.arange(rows, dtype=np.float32))
        valid = flow[..., -1].astype(bool)
        assert flow.shape[:-1] == depth_gt_0.shape == depth_gt_1.shape
        mask = mask_0 & valid
        x_1 = x + flow[..., 0]
        y_1 = y + flow[..., 1]

        points_0 = np.stack([x, y, depth_gt_0])
        points_1 = np.stack([x_1, y_1, depth_gt_1])

        points_1[:, occ_mask] = points_0[:, occ_mask]
        points_0 = points_0.reshape((3, -1)).T
        points_1 = points_1.reshape((3, -1)).T
        points_0 = points_0[mask.reshape(-1)]
        points_1 = points_1[mask.reshape(-1)]

        cloud_0 = self.project_image_to_velo(points_0)
        cloud_1 = self.project_image_to_velo(points_1)
        flow = cloud_1 - cloud_0

        return cloud_0, cloud_1, flow

    def __len__(self):
        return len(self.left_0_filenames)

    def __getitem__(self, index):
        left_img_0_ = self.load_image(os.path.join(self.datapath, self.left_0_filenames[index]))
        right_img_0_ = self.load_image(os.path.join(self.datapath, self.right_0_filenames[index]))
        left_img_1_ = self.load_image(os.path.join(self.datapath, self.left_1_filenames[index]))
        right_img_1_ = self.load_image(os.path.join(self.datapath, self.right_1_filenames[index]))
        T_world_cam_101, cam_101, T_world_cam_103, cam_103 = self.load_calib(
            os.path.join(self.datapath, self.calib_filenames[index]))
        disp_gt_0 = None
        disp_gt_1 = None
        op_flow_gt = None
        if arrays_not_none(self.disp_0_filenames, self.disp_1_filenames, self.flow_filenames):
            disp_gt_0 = self.load_disp(os.path.join(self.datapath, self.disp_0_filenames[index]))
            disp_gt_1 = self.load_disp(os.path.join(self.datapath, self.disp_1_filenames[index]))
            op_flow_gt = self.load_flow(os.path.join(self.datapath, self.flow_filenames[index]))

        # numpy to tensor
        T_world_cam_101 = torch.from_numpy(T_world_cam_101)
        T_world_cam_103 = torch.from_numpy(T_world_cam_103)

        w, h = left_img_0_.size
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
                left_img_0 = Image.new(left_img_0_.mode, (crop_w, crop_h), (0, 0, 0))
                left_img_0.paste(left_img_0_, (0, 0))
                right_img_0 = Image.new(right_img_0_.mode, (crop_w, crop_h), (0, 0, 0))
                right_img_0.paste(right_img_0_, (0, 0))
                left_img_1 = Image.new(left_img_1_.mode, (crop_w, crop_h), (0, 0, 0))
                left_img_1.paste(left_img_1_, (0, 0))
                right_img_1 = Image.new(right_img_1_.mode, (crop_w, crop_h), (0, 0, 0))
                right_img_1.paste(right_img_1_, (0, 0))
                if arrays_not_none(disp_gt_0, disp_gt_1, op_flow_gt):
                    disp_gt_0 = np.lib.pad(
                        disp_gt_0, ((0, 0), (h_pad, w_pad)), mode='constant', constant_values=0)
                    disp_gt_1 = np.lib.pad(
                        disp_gt_1, ((0, 0), (h_pad, w_pad)), mode='constant', constant_values=0)
                    op_flow_gt = np.lib.pad(
                        op_flow_gt, ((0, 0), (h_pad, w_pad), (0, 0)), mode='constant', constant_values=0)
            else:
                w_crop = w - crop_w
                h_crop = h - crop_h
                left_img_0 = left_img_0_.crop((w_crop, h_crop, w, h))
                right_img_0 = right_img_0_.crop((w_crop, h_crop, w, h))
                left_img_1 = left_img_1_.crop((w_crop, h_crop, w, h))
                right_img_1 = right_img_1_.crop((w_crop, h_crop, w, h))
                if arrays_not_none(disp_gt_0, disp_gt_1, op_flow_gt):
                    disp_gt_0 = disp_gt_0[h_crop: h, w_crop: w]
                    disp_gt_1 = disp_gt_1[h_crop: h, w_crop: w]
                    op_flow_gt = op_flow_gt[h_crop: h, w_crop: w]

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

        filtered_cloud_gt_0 = None
        filtered_cloud_gt_1 = None
        all_vox_grid_0_gt = []
        all_vox_grid_1_gt = []
        vox_flow_gt = None
        if arrays_not_none(disp_gt_0, disp_gt_1, op_flow_gt):
            cloud_gt_0, cloud_gt_1, flow_gt = self.calc_cloud_flow(disp_gt_0, disp_gt_1, op_flow_gt)
            cloud_gt_0 = np.concatenate([cloud_gt_0, flow_gt], axis=-1)

            filtered_cloud_gt_0 = self.filter_cloud(cloud_gt_0)
            filtered_cloud_gt_1 = self.filter_cloud(cloud_gt_1)

            if self.stored_gt:
                all_vox_grid_0_gt = self.load_gt(os.path.join(self.datapath, self.gt_voxel_0_filenames[index]))
                all_vox_grid_1_gt = self.load_gt(os.path.join(self.datapath, self.gt_voxel_1_filenames[index]))
                vox_flow_gt = self.load_gt(os.path.join(self.datapath, self.gt_flow_filenames[index]))
                valid_gt_0, _ = self.calc_voxel_grid(filtered_cloud_gt_0, 0)
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
        cam_101 = Camera(np.concatenate(([imw, imh], cam_101)).astype(np.float32))
        cam_101 = cam_101.crop(left_top, torch.tensor([crop_w, crop_h]))
        cam_101 = cam_101.scale(scale)
        cam_103 = Camera(np.concatenate(([imw, imh], cam_103)).astype(np.float32))
        cam_103 = cam_103.crop(left_top, torch.tensor([crop_w, crop_h]))
        cam_103 = cam_103.scale(scale)

        return {'left_0': left_img_0,
                'right_0': right_img_0,
                'left_1': left_img_1,
                'right_1': right_img_1,
                'T_world_cam_101_0': T_world_cam_101,
                'cam_101_0': cam_101.data,
                'T_world_cam_103_0': T_world_cam_103,
                'cam_103_0': cam_103.data,
                'T_world_cam_101_1': T_world_cam_101,
                'cam_101_1': cam_101.data,
                'T_world_cam_103_1': T_world_cam_103,
                'cam_103_1': cam_103.data,
                'voxel_grid_0': all_vox_grid_0_gt if len(all_vox_grid_0_gt) >= 0 else 'null',
                'voxel_grid_1': all_vox_grid_1_gt if len(all_vox_grid_1_gt) >= 0 else 'null',
                'voxel_flow': vox_flow_gt if vox_flow_gt is not None else 'null',
                'point_cloud_0': filtered_cloud_gt_0.astype(
                    np.float32).tobytes() if filtered_cloud_gt_0 is not None else 'null',
                'point_cloud_1': filtered_cloud_gt_1.astype(
                    np.float32).tobytes() if filtered_cloud_gt_1 is not None else 'null',
                "left_filename": self.left_0_filenames[index]}
