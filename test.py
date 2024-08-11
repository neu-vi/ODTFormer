import hydra
import torch.backends.cudnn as cudnn
import torch.cuda
from omegaconf import DictConfig
from pytorch3d.loss import chamfer_distance
from torch.utils.data import DataLoader
from tqdm import tqdm
from thop import profile, clever_format
import time

from models import calc_IoU, eval_metric
from utils import *

logger = logging.getLogger(__name__)
coloredlogs.install(level='INFO')

BATCH_SIZE = 1

cudnn.benchmark = True


@hydra.main(version_base=None, config_path='./config', config_name='eval_model')
def eval_model(cfg: DictConfig):
    model = hydra.utils.instantiate(cfg.model, _recursive_=False)
    if torch.cuda.is_available():
        model.cuda()

    test_dataset = hydra.utils.instantiate(cfg.dataloader.dataset.test_dataset)
    TestImgLoader = DataLoader(
        test_dataset, BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True, drop_last=False, prefetch_factor=4)

    state_dict = torch.load(cfg.ckpt_path)['model']
    new_state_dict = {}
    for k, v in state_dict.items():
        k = k[7:]
        new_state_dict[k] = v

    model.load_state_dict(new_state_dict, strict=True)
    # model = torch.compile(model)
    model.eval()

    iou_dict = MetricDict()
    cd_dict = MetricDict()
    infer_time = []
    for batch_idx, sample in enumerate(tqdm(TestImgLoader)):
        imgL, imgR, voxel_gt = sample['left'], sample['right'], sample['voxel_grid']
        calib_meta = {'T_world_cam_101': sample['T_world_cam_101'], 'T_world_cam_103': sample['T_world_cam_103'],
                      'cam_101': sample['cam_101'], 'cam_103': sample['cam_103']}

        if torch.cuda.is_available():
            imgL = imgL.cuda()
            imgR = imgR.cuda()
            for i in range(len(voxel_gt)):
                voxel_gt[i] = voxel_gt[i].cuda()

        torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            voxel_ests = model(imgL, imgR, calib_meta=calib_meta)
            torch.cuda.synchronize()
            infer_time.append(time.time() - start)
            for voxel_pred in voxel_ests:
                voxel_pred[voxel_pred < 0.5] = 0
                voxel_pred[voxel_pred >= 0.5] = 1
        iou_dict.append(eval_metric(voxel_ests, voxel_gt, calc_IoU, depth_range=cfg.depth_range))
        cd_dict.append(eval_metric(voxel_ests, voxel_gt, eval_cd, cfg.dataloader.vox, depth_range=cfg.depth_range))

    iou_mean = iou_dict.mean()
    cd_mean = cd_dict.mean()

    for k in iou_mean.keys():
        msg = f'Depth - {k}: IoU = {str(iou_mean[k].tolist())}; CD = {str(cd_mean[k].tolist())}'
        logger.info(msg)
    avg_infer = np.mean(np.array(infer_time))
    logger.info(f'Avg_infer = {avg_infer}; FPS = {1 / avg_infer}')


def eval_cd(pred, gt, scale):
    pred_coord = torch.nonzero((pred.squeeze(0) >= 0.5).int()) * float(scale)
    gt_coord = torch.nonzero((gt.squeeze(0) == 1).int()) * float(scale)

    return chamfer_distance(pred_coord[None, ...], gt_coord[None, ...])[0]


class MetricDict:
    def __init__(self):
        self._data = {}

    def append(self, in_dict):
        for k, v, in in_dict.items():
            if k not in self._data:
                self._data[k] = [v]
            else:
                self._data[k].append(v)

    def mean(self):
        out_dict = {}
        for k, v in self._data.items():
            v_t = torch.asarray(v)
            out_dict[k] = torch.mean(v_t, dim=0)

        return out_dict

    def __getattr__(self, item):
        return getattr(self._data, item)()

    def __getitem__(self, item):
        return self._data[item]


@hydra.main(version_base=None, config_path='./config', config_name='eval_model')
def eval_ops(cfg: DictConfig):
    model = hydra.utils.instantiate(cfg.model, _recursive_=False)
    if torch.cuda.is_available():
        model.cuda()

    test_dataset = hydra.utils.instantiate(cfg.dataloader.dataset.test_dataset)
    subset = torch.utils.data.Subset(test_dataset, [0])
    EvalImgLoader = DataLoader(subset, BATCH_SIZE, shuffle=False)

    for sample in EvalImgLoader:
        imgL, imgR, voxel_gt = sample['left'], sample['right'], sample['voxel_grid']
        calib_meta = {'T_world_cam_101': sample['T_world_cam_101'], 'T_world_cam_103': sample['T_world_cam_103'],
                      'cam_101': sample['cam_101'], 'cam_103': sample['cam_103']}
        if torch.cuda.is_available():
            imgL = imgL.cuda()
            imgR = imgR.cuda()
            for i in range(len(voxel_gt)):
                voxel_gt[i] = voxel_gt[i].cuda()

        with torch.no_grad():
            macs, params = clever_format(profile(model, inputs=(imgL, imgR, calib_meta)), '%.3f')

    print(f'MACS: {macs}, PARAMS: {params}')


if __name__ == '__main__':
    eval_model()
    # eval_ops()
