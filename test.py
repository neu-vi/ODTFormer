import hydra
import torch.backends.cudnn as cudnn
import torch.cuda
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm
from thop import profile, clever_format

from models import VoxelFlow, calc_IoU, scene_epe, foreground_epe
from utils import *

logger = logging.getLogger(__name__)
coloredlogs.install(level='INFO')

BATCH_SIZE = 1
ROI_SCALE = [-9, 9, -3, 3, 0, 30]
VOX_SIZES = [3, 1.5, 0.75, 0.375]

cudnn.benchmark = True


@hydra.main(version_base=None, config_path='./config', config_name='eval_model')
def eval_model(cfg: DictConfig):
    model = VoxelFlow(cfg.model, ROI_SCALE, VOX_SIZES[-1], (9, 3, 9))
    if torch.cuda.is_available():
        model.cuda()

    test_dataset = hydra.utils.instantiate(cfg.dataloader.dataset.test_dataset)
    TestImgLoader = DataLoader(
        test_dataset, BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True, drop_last=False,
        persistent_workers=True, prefetch_factor=4)

    state_dict = torch.load(cfg.ckpt_path)['model']
    new_state_dict = {}
    for k, v in state_dict.items():
        k = k[7:]
        new_state_dict[k] = v

    model.load_state_dict(new_state_dict, strict=True)
    model.eval()

    # iou_l = []
    epe_s_l = []
    epe_f_l = []
    for batch_idx, sample in enumerate(tqdm(TestImgLoader)):
        imgL_0, imgR_0 = sample['left_0'], sample['right_0']
        imgL_1, imgR_1 = sample['left_1'], sample['right_1']
        voxel_gt_0, voxel_gt_1 = sample['voxel_grid_0'][-1], sample['voxel_grid_1'][-1]
        flow_gt = sample['voxel_flow']
        calib_meta_0 = {'T_world_cam_101': sample['T_world_cam_101_0'], 'T_world_cam_103': sample['T_world_cam_103_0'],
                        'cam_101': sample['cam_101_0'], 'cam_103': sample['cam_103_0']}
        calib_meta_1 = {'T_world_cam_101': sample['T_world_cam_101_1'], 'T_world_cam_103': sample['T_world_cam_103_1'],
                        'cam_101': sample['cam_101_1'], 'cam_103': sample['cam_103_1']}

        if torch.cuda.is_available():
            imgL_0 = imgL_0.cuda()
            imgR_0 = imgR_0.cuda()
            imgL_1 = imgL_1.cuda()
            imgR_1 = imgR_1.cuda()
            flow_gt = flow_gt.cuda()
            voxel_gt_0 = voxel_gt_0.cuda()
            voxel_gt_1 = voxel_gt_1.cuda()

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
        # iou_l.append(calc_IoU(voxel_est_0[-1], voxel_gt_0).item())
        # iou_l.append(calc_IoU(voxel_est_1[-1], voxel_gt_1).item())
        epe_s_l.append(scene_epe(flow_est, flow_gt).item())
        epe_f_l.append(foreground_epe(flow_est, flow_gt, voxel_gt_0).item())

    # iou_l = np.array(iou_l)
    epe_s_l = np.array(epe_s_l)
    epe_f_l = np.array(epe_f_l)

    logger.info(f'Scene EPE: {np.mean(epe_s_l)}; Foreground EPE: {np.mean(epe_f_l)}')


@hydra.main(version_base=None, config_path='./config', config_name='eval_model')
def eval_ops(cfg: DictConfig):
    model = VoxelFlow(cfg.model, ROI_SCALE, VOX_SIZES[-1], (9, 3, 9))
    if torch.cuda.is_available():
        model.cuda()

    test_dataset = hydra.utils.instantiate(cfg.dataloader.dataset.test_dataset)
    subset = torch.utils.data.Subset(test_dataset, [0])
    EvalImgLoader = DataLoader(subset, BATCH_SIZE, shuffle=False)

    for sample in EvalImgLoader:
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
            macs, params = clever_format(
                profile(model, inputs=((imgL_0, imgR_0), (imgL_1, imgR_1), False, pair_0_kwargs, pair_1_kwargs)),
                '%.3f')

            print(f'MACS: {macs}, PARAMS: {params}')


if __name__ == '__main__':
    eval_model()
    # eval_ops()
