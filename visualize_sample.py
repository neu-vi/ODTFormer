import hydra
import torch.backends.cudnn as cudnn
import torch.cuda
from omegaconf import DictConfig
import numpy as np
import os

SAVE_DIR = './visualization/demo/'

cudnn.benchmark = True


@hydra.main(version_base=None, config_path='./config', config_name='eval_model')
def main(cfg: DictConfig):
    model = hydra.utils.instantiate(cfg.model, _recursive_=False)
    if torch.cuda.is_available():
        model.cuda()

    sample_dataset = hydra.utils.instantiate(cfg.dataloader.dataset.test_dataset)
    state_dict = torch.load(cfg.ckpt_path)['model']
    new_state_dict = {}
    for k, v in state_dict.items():
        k = k[7:]
        new_state_dict[k] = v

    model.load_state_dict(new_state_dict, strict=True)
    model.eval()

    for idx in cfg.sample_indices:
        file_id = f'{idx:06d}'

        imgL = sample_dataset[idx]['left'][None, ...]
        imgR = sample_dataset[idx]['right'][None, ...]
        colored_cloud_gt = torch.from_numpy(
            np.frombuffer(sample_dataset[idx]['point_cloud'], dtype=np.float32).reshape(-1, 6))
        calib_meta = {'T_world_cam_101': sample_dataset[idx]['T_world_cam_101'][None, ...],
                      'T_world_cam_103': sample_dataset[idx]['T_world_cam_103'][None, ...],
                      'cam_101': torch.from_numpy(sample_dataset[idx]['cam_101'])[None, ...],
                      'cam_103': torch.from_numpy(sample_dataset[idx]['cam_103'])[None, ...],
                      'left_top': torch.from_numpy(sample_dataset[idx]['left_top'])[None, ...]}
        '''
        # visualize ground truth labels
        voxel_gt = sample_dataset[idx]['voxel_grid']
        np.savez(os.path.join(SAVE_DIR, f'gt_{file_id}.npz'), colored_pc_gt=colored_cloud_gt,
                 level_0=voxel_gt[0].numpy(), level_1=voxel_gt[1].numpy(), level_2=voxel_gt[2].numpy(),
                 level_3=voxel_gt[3].numpy())
        '''

        if torch.cuda.is_available():
            imgL = imgL.cuda()
            imgR = imgR.cuda()

        with torch.no_grad():
            voxel_ests = model(imgL, imgR, calib_meta=calib_meta, training=False)
            for voxel_pred in voxel_ests:
                voxel_pred[voxel_pred < 0.5] = 0
                voxel_pred[voxel_pred >= 0.5] = 1

        if not os.path.exists(SAVE_DIR):
            os.makedirs(SAVE_DIR)
        np.savez(os.path.join(SAVE_DIR, f'vox_{file_id}.npz'), colored_pc_gt=colored_cloud_gt,
                 level_0=voxel_ests[0].cpu().numpy(), level_1=voxel_ests[1].cpu().numpy(),
                 level_2=voxel_ests[2].cpu().numpy(), level_3=voxel_ests[3].cpu().numpy())


if __name__ == '__main__':
    main()
