import os.path

import hydra
import torch.backends.cudnn as cudnn
import torch.cuda
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import VoxelFlow
from datasets import KITTIVoxelFLow
from utils import *

logger = logging.getLogger(__name__)
coloredlogs.install(level='INFO')

BATCH_SIZE = 1
ROI_SCALE = [-9, 9, -3, 3, 0, 30]
VOX_SIZES = [3, 1.5, 0.75, 0.375]

cudnn.benchmark = True
SAVE_DIR = './visualization/demo/'


@hydra.main(version_base=None, config_path='./config', config_name='eval_model')
def store_sample(cfg: DictConfig):
    model = VoxelFlow(cfg.model, ROI_SCALE, VOX_SIZES[-1], (21, 3, 21))
    if torch.cuda.is_available():
        model.cuda()

    viz_dataset = KITTIVoxelFLow('.../KITTI_VoxelFlow', './filenames/KITTI_flow_viz.txt', False,
                                 [-9, 9, -3, 3, 0, 30], [3, 1.5, 0.75, 0.375], resize_shape=[400, 880])
    TestImgLoader = DataLoader(
        viz_dataset, BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True, drop_last=False,
        persistent_workers=True, prefetch_factor=4)

    state_dict = torch.load(cfg.ckpt_path)['model']
    new_state_dict = {}
    for k, v in state_dict.items():
        k = k[7:]
        new_state_dict[k] = v

    model.load_state_dict(new_state_dict, strict=True)
    model.eval()

    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    for batch_idx, sample in enumerate(tqdm(TestImgLoader)):
        left_filename = sample['left_filename'][0]
        file_id = left_filename.split('/')[-1].split('_')[0]

        imgL_0, imgR_0 = sample['left_0'], sample['right_0']
        imgL_1, imgR_1 = sample['left_1'], sample['right_1']
        calib_meta_0 = {'T_world_cam_101': sample['T_world_cam_101_0'], 'T_world_cam_103': sample['T_world_cam_103_0'],
                        'cam_101': sample['cam_101_0'], 'cam_103': sample['cam_103_0']}
        calib_meta_1 = {'T_world_cam_101': sample['T_world_cam_101_1'], 'T_world_cam_103': sample['T_world_cam_103_1'],
                        'cam_101': sample['cam_101_1'], 'cam_103': sample['cam_103_1']}

        if torch.cuda.is_available():
            imgL_0 = imgL_0.cuda()
            imgR_0 = imgR_0.cuda()
            imgL_1 = imgL_1.cuda()
            imgR_1 = imgR_1.cuda()

        pair_0_kwargs = {'calib_meta': calib_meta_0}
        pair_1_kwargs = {'calib_meta': calib_meta_1}
        with torch.no_grad():
            voxel_ests, flow_est = model((imgL_0, imgR_0), (imgL_1, imgR_1), training=False,
                                         pair_0_kwargs=pair_0_kwargs, pair_1_kwargs=pair_1_kwargs)

        voxel_est_0, voxel_est_1 = voxel_ests
        for voxel_pred in voxel_est_0:
            voxel_pred[voxel_pred < 0.5] = 0
            voxel_pred[voxel_pred >= 0.5] = 1
        for voxel_pred in voxel_est_1:
            voxel_pred[voxel_pred < 0.5] = 0
            voxel_pred[voxel_pred >= 0.5] = 1

        np.savez(os.path.join(SAVE_DIR, f'{file_id}.npz'), voxel_0=voxel_est_0[3][0].cpu().numpy(),
                 voxel_1=voxel_est_1[3][0].cpu().numpy(), flow=flow_est[0].cpu().numpy())


if __name__ == '__main__':
    store_sample()
