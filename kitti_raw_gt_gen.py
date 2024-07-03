import os
import sys
import torch
from tqdm import tqdm
import numpy as np
from datasets import *
from models.wrappers import Camera

root = '/work/vig/Datasets/KITTI_VoxelFlow'


def get_gt_voxel(filename, max_array=1, array_id=1):
    assert array_id <= max_array

    dataloader = VoxelKITTIRaw(root, filename, True, [-9, 9, -3, 3, 0, 30], [3, 1.5, 0.75, 0.375])
    total_files = len(dataloader.gt_voxel_filenames)

    partition_size = total_files // max_array
    right_bound = array_id * partition_size if array_id < max_array else total_files
    for idx in tqdm(range((array_id - 1) * partition_size, right_bound), 'Generating gt labels'):
        splits_vox = dataloader.gt_voxel_filenames[idx].split('/')
        dirs_vox = os.path.join(root, '/'.join(splits_vox[:-1]))

        if not os.path.exists(dirs_vox):
            try:
                os.makedirs(dirs_vox)
            except Exception as e:
                pass

        _, cam_101, _, _ = dataloader.load_calib(os.path.join(root, dataloader.calib_filenames[idx]))
        dataloader.load_velo2cam(os.path.join(root, dataloader.velo2cam_filenames[idx]))
        scan_gt = dataloader.load_velo(os.path.join(root, dataloader.velo_filenames[idx]))
        ###############################################################
        crop_w, crop_h = 1224, 370
        cam_101 = np.concatenate(([crop_w, crop_h], cam_101)).astype(np.float32)
        ###############################################################
        scan_cam_101 = dataloader.lidar_extrinsic.transform(scan_gt).numpy()
        valid_scan = Camera(torch.from_numpy(cam_101)).project(scan_cam_101)[1]
        cloud_gt = scan_cam_101[valid_scan]
        filtered_cloud_gt = dataloader.filter_cloud(cloud_gt)

        all_vox_grid_gt = []
        parent_grid = None

        for level in range(len(dataloader.grid_sizes)):
            vox_grid_gt, cloud_np_gt = dataloader.calc_voxel_grid(
                filtered_cloud_gt, level=level, parent_grid=parent_grid)
            vox_grid_gt = torch.from_numpy(vox_grid_gt)

            parent_grid = vox_grid_gt
            all_vox_grid_gt.append(vox_grid_gt)

        torch.save(all_vox_grid_gt, os.path.join(root, dataloader.gt_voxel_filenames[idx]))


if __name__ == '__main__':
    # get_gt_voxel('./filenames/DS_test_gt_calib.txt')
    # array_id = os.getenv("SLURM_ARRAY_TASK_ID")
    get_gt_voxel(
        '/work/vig/tianyed/StereoVoxelFormer/stereo-voxel-former/scripts/net/filenames/KITTI_raw.txt', 50,
        int(sys.argv[1]))
