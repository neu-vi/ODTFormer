import os
import sys
import torch
from tqdm import tqdm
import numpy as np
from datasets import KITTIVoxelFLow
from datasets.data_io import get_transform

root = '.../KITTI_VoxelFlow'


def get_gt_voxel(filename, max_array=1, array_id=1):
    assert array_id <= max_array

    dataloader = KITTIVoxelFLow(root, filename, True, [-9, 9, -3, 3, 0, 30], [3, 1.5, 0.75, 0.375])
    assert len(dataloader.gt_voxel_0_filenames) == len(dataloader.gt_voxel_1_filenames)
    total_files = len(dataloader.gt_voxel_0_filenames)

    partition_size = total_files // max_array
    right_bound = array_id * partition_size if array_id < max_array else total_files
    for idx in tqdm(range((array_id - 1) * partition_size, right_bound), 'Generating gt labels'):
        splits_0 = dataloader.gt_voxel_0_filenames[idx].split('/')
        splits_1 = dataloader.gt_voxel_1_filenames[idx].split('/')
        splits_flow = dataloader.gt_flow_filenames[idx].split('/')
        dirs_0 = os.path.join(root, '/'.join(splits_0[:-1]))
        dirs_1 = os.path.join(root, '/'.join(splits_1[:-1]))
        dirs_flow = os.path.join(root, '/'.join(splits_flow[:-1]))

        if not os.path.exists(dirs_0):
            try:
                os.makedirs(dirs_0)
            except Exception as e:
                pass
        if not os.path.exists(dirs_1):
            try:
                os.makedirs(dirs_1)
            except Exception as e:
                pass
        if not os.path.exists(dirs_flow):
            try:
                os.makedirs(dirs_flow)
            except Exception as e:
                pass

        calib_ = dataloader.load_calib(os.path.join(root, dataloader.calib_filenames[idx]))
        disp_gt_0 = dataloader.load_disp(os.path.join(root, dataloader.disp_0_filenames[idx]))
        disp_gt_1 = dataloader.load_disp(os.path.join(root, dataloader.disp_1_filenames[idx]))
        op_flow_gt = dataloader.load_flow(os.path.join(root, dataloader.flow_filenames[idx]))
        ###############################################################
        left_img = dataloader.load_image(os.path.join(root, dataloader.left_0_filenames[idx]))
        w, h = left_img.size
        crop_w, crop_h = 1224, 370

        processed = get_transform(dataloader.color_jitter)
        left_top = [0, 0]
        if dataloader.transform:
            if w < crop_w:
                w_pad = crop_w - w
                disp_gt_0 = np.lib.pad(
                    disp_gt_0, ((0, 0), (0, w_pad)), mode='constant', constant_values=0)
                disp_gt_1 = np.lib.pad(
                    disp_gt_1, ((0, 0), (0, w_pad)), mode='constant', constant_values=0)
                op_flow_gt = np.lib.pad(
                    op_flow_gt, ((0, 0), (0, w_pad), (0, 0)), mode='constant', constant_values=0)
            else:
                w_crop = w - crop_w
                h_crop = h - crop_h
                disp_gt_0 = disp_gt_0[h_crop: h, w_crop: w]
                disp_gt_1 = disp_gt_1[h_crop: h, w_crop: w]
                op_flow_gt = op_flow_gt[h_crop: h, w_crop: w]

                left_top = [w_crop, h_crop]
        else:
            w_crop = w - crop_w
            h_crop = h - crop_h
        ###############################################################
        cloud_gt_0, cloud_gt_1, flow_gt = dataloader.calc_cloud_flow(disp_gt_0, disp_gt_1, op_flow_gt)
        cloud_gt_0 = np.concatenate([cloud_gt_0, flow_gt], axis=-1)

        filtered_cloud_gt_0 = dataloader.filter_cloud(cloud_gt_0)
        filtered_cloud_gt_1 = dataloader.filter_cloud(cloud_gt_1)
        all_vox_grid_0_gt = []
        all_vox_grid_1_gt = []
        parent_grid_0 = None
        parent_grid_1 = None
        vox_flow_gt = None

        for level in range(len(dataloader.grid_sizes)):
            occupied_gate = dataloader.occupied_gates[level]
            if level == len(dataloader.grid_sizes) - 1:
                vox_grid_gt_0, cloud_np_gt_0, vox_flow_gt = dataloader.calc_voxel_grid(
                    filtered_cloud_gt_0, level=level, parent_grid=parent_grid_0, occupied_gate=occupied_gate,
                    get_flow=True)
            else:
                vox_grid_gt_0, cloud_np_gt_0 = dataloader.calc_voxel_grid(
                    filtered_cloud_gt_0, level=level, parent_grid=parent_grid_0, occupied_gate=occupied_gate)

            vox_grid_gt_1, cloud_np_gt_1 = dataloader.calc_voxel_grid(
                filtered_cloud_gt_1, level=level, parent_grid=parent_grid_1, occupied_gate=occupied_gate)
            vox_grid_gt_0 = torch.from_numpy(vox_grid_gt_0)
            vox_grid_gt_1 = torch.from_numpy(vox_grid_gt_1)

            parent_grid_0 = vox_grid_gt_0
            parent_grid_1 = vox_grid_gt_1
            all_vox_grid_0_gt.append(vox_grid_gt_0)
            all_vox_grid_1_gt.append(vox_grid_gt_1)

        torch.save(all_vox_grid_0_gt, os.path.join(root, dataloader.gt_voxel_0_filenames[idx]))
        torch.save(all_vox_grid_1_gt, os.path.join(root, dataloader.gt_voxel_1_filenames[idx]))
        torch.save(torch.from_numpy(vox_flow_gt), os.path.join(root, dataloader.gt_flow_filenames[idx]))


if __name__ == '__main__':
    # array_id = os.getenv("SLURM_ARRAY_TASK_ID")
    get_gt_voxel('./filenames/KITTI_flow.txt', 50,
                 int(sys.argv[1]))
