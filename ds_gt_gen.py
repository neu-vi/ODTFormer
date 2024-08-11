import os
import sys
import torch
from tqdm import tqdm
import numpy as np

from datasets import VoxelDSDatasetCalib

root = '/work/vig/Datasets/DrivingStereo'


def get_gt_voxel(filename, max_array=1, array_id=1):
    assert array_id <= max_array

    dataloader = VoxelDSDatasetCalib(root, filename, False, [-8, 10, -3, 3, 0, 30], [3, 1.5, 0.75, 0.375])
    assert len(dataloader.gt_filenames) == len(dataloader.depth_filenames) == len(dataloader.calib_filenames)
    total_files = len(dataloader.gt_filenames)

    partition_size = total_files // max_array
    right_bound = array_id * partition_size if array_id < max_array else total_files

    for idx in tqdm(range((array_id - 1) * partition_size, right_bound), 'Generating gt labels'):
        splits = dataloader.gt_filenames[idx].split('/')
        dirs = os.path.join(root, '/'.join(splits[:-1]))

        if not os.path.exists(dirs):
            try:
                os.mkdir(dirs)
            except Exception as e:
                pass

        calib_ = dataloader.load_calib(os.path.join(root, dataloader.calib_filenames[idx]))
        depth_gt = dataloader.load_depth(os.path.join(root, dataloader.depth_filenames[idx]))
        ###############################################################
        left_img = dataloader.load_image(os.path.join(root, dataloader.left_filenames[idx]))
        w, h = left_img.size
        crop_w, crop_h = 880, 400

        left_top = [0, 0]

        if dataloader.transform:
            if w < crop_w:
                w_pad = crop_w - w
                depth_gt = np.lib.pad(
                    depth_gt, ((0, 0), (0, w_pad)), mode='constant', constant_values=0)
            else:
                w_crop = w - crop_w
                h_crop = h - crop_h
                depth_gt = depth_gt[h_crop: h, w_crop: w]

                left_top = [w_crop, h_crop]
        else:
            w_crop = w - crop_w
            h_crop = h - crop_h
        ###############################################################
        cloud_gt = dataloader.calc_cloud(depth_gt)
        filtered_cloud_gt = dataloader.filter_cloud(cloud_gt)

        all_vox_grid_gt = []
        parent_grid = None
        try:
            for level in range(len(dataloader.grid_sizes)):
                vox_grid_gt, cloud_np_gt = dataloader.calc_voxel_grid(
                    filtered_cloud_gt, level=level, parent_grid=parent_grid)
                vox_grid_gt = torch.from_numpy(vox_grid_gt)

                parent_grid = vox_grid_gt
                all_vox_grid_gt.append(vox_grid_gt)
        except Exception as e:
            raise RuntimeError('Error in calculating voxel grids from point cloud')

        torch.save(all_vox_grid_gt, os.path.join(root, dataloader.gt_filenames[idx]))


if __name__ == '__main__':
    # get_gt_voxel('./filenames/DS_test_gt_calib.txt')
    # array_id = os.getenv("SLURM_ARRAY_TASK_ID")
    get_gt_voxel('/work/vig/tianyed/StereoVoxelFormer/stereo-voxel-former/scripts/net/filenames/DS_train_gt_calib.txt',
                 50, int(sys.argv[1]))
