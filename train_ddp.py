from __future__ import print_function, division
import sys
import os
import gc
import time
# import torch.nn as nn
import torch.utils.data
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data.distributed
import torch.backends.cudnn as cudnn
# import torch.distributed as dist
import torch.multiprocessing as mp

from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from models import VoxelFlow, model_loss, model_iou, scene_epe, foreground_epe
from utils import *
import logging
import coloredlogs
import wandb
import hydra
from omegaconf import DictConfig

import utils

cudnn.benchmark = True
log = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG', logger=log)

torch.autograd.set_detect_anomaly(False)


# scaler = torch.cuda.amp.GradScaler(growth_interval=100)


@hydra.main(version_base=None, config_path='./config', config_name='train')
def main(cfg: DictConfig):
    # parse arguments, set seeds

    # torch.manual_seed(args.seed)
    torch.manual_seed(cfg.trainer.seed)
    torch.cuda.manual_seed(cfg.trainer.seed)

    if cfg.dist.dist_url == "env://" and cfg.dist.world_size == -1:
        cfg.dist.world_size = int(os.environ["WORLD_SIZE"])

    cfg.trainer.distributed = cfg.dist.world_size > 1 or cfg.dist.multiprocessing_distributed

    if torch.cuda.is_available():
        ngpus_per_node = torch.cuda.device_count()
    else:
        ngpus_per_node = 1

    config = {
        "lr": cfg.optimizer.lr,
        "bs": cfg.dataloader.batch_size,
        'loss': cfg.trainer.loss,
        'bbone': cfg.model.backbone.name,
        'bblayer': cfg.model.backbone.layer,
        'attn': cfg.model.decoder_layer.attn_name
    }

    if cfg.dist.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        cfg.dist.world_size = ngpus_per_node * cfg.dist.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, cfg, config))
    else:
        # Simply call main_worker function
        # main_worker(cfg.dist.gpu, ngpus_per_node, cfg, config)
        main_worker(0, ngpus_per_node, cfg, config)


def main_worker(gpu, ngpus_per_node, cfg, config=None):
    cfg.dist.gpu = gpu
    log.info('Using GPU: {}'.format(cfg.dist.gpu))

    if cfg.trainer.distributed:
        if cfg.dist.dist_url == "env://" and cfg.dist.rank == -1:
            cfg.dist.rank = int(os.environ["RANK"])
        if cfg.dist.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            cfg.dist.rank = cfg.dist.rank * ngpus_per_node + gpu
        dist_url = cfg.dist.dist_url + ':' + str(cfg.dist.port)
        dist.init_process_group(backend=cfg.dist.dist_backend, init_method=dist_url,
                                world_size=cfg.dist.world_size, rank=cfg.dist.rank)

    if utils.is_main_process():
        log.info("===========" + cfg.model._target_.split('.')[-1] + "===========")

    log_info = ""

    for ix in range(1, len(sys.argv)):
        if not (sys.argv[ix].startswith('dist') or sys.argv[ix].startswith('trainer.logdir')):
            if ix == len(sys.argv) - 1:
                log_info += '{}'.format(sys.argv[ix])
            else:
                log_info += '{}_'.format(sys.argv[ix])
    log_info = log_info.replace('/', '.')
    log_info = log_info.replace('=', '#')
    log_info += '\n'

    cfg.trainer.loss_weights = [float(item) for item in cfg.trainer.loss_weights.split(',')]

    if utils.is_main_process():
        logdir = os.path.join(cfg.trainer.logdir, cfg.trainer.logdir_name) + '#'
        counter = 0
        while os.path.exists(logdir + str(counter) + '/'):
            counter += 1

        logdir += str(counter)
        cfg.trainer.logdir = logdir
        os.makedirs(cfg.trainer.logdir, mode=0o770, exist_ok=True)
        with open(cfg.trainer.logdir + '/setting.info', 'w+') as f:
            f.write(log_info)

        # create summary logger
        logger = SummaryWriter(cfg.trainer.logdir)

    model = VoxelFlow(cfg.model, cfg.dataloader.roi, cfg.dataloader.vox[-1], (9, 3, 9))
    if utils.is_main_process():
        log.info(model)
        log.info('Number of parameters: {:.6f}M'.format(sum([p.data.nelement() for p in model.parameters()]) / 1000000))
        log.info('loss weights: {}'.format(cfg.trainer.loss_weights))
        log.info(f"Saving log at directory {cfg.trainer.logdir}")
    if cfg.trainer.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if torch.cuda.is_available():
            if cfg.dist.gpu is not None:
                torch.cuda.set_device(cfg.dist.gpu)
                model.cuda(cfg.dist.gpu)
                # When using a single GPU per process and per
                # DistributedDataParallel, we need to divide the batch size
                # ourselves based on the total number of GPUs of the current node.
                cfg.dataloader.batch_size = int(cfg.dataloader.batch_size / ngpus_per_node)
                cfg.dataloader.test_batch_size = 1  # max(1, int(cfg.dataloader.test_batch_size / ngpus_per_node))
                cfg.dataloader.workers = int((cfg.dataloader.workers + ngpus_per_node - 1) / ngpus_per_node)
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[cfg.dist.gpu],
                                                                  find_unused_parameters=True)
            else:
                model.cuda()
                # DistributedDataParallel will divide and allocate batch_size to all
                # available GPUs if device_ids are not set
                model = torch.nn.parallel.DistributedDataParallel(model)
        #  model = torch.compile(model)
    else:
        model = torch.nn.DataParallel(model).cuda()

    device = 'cpu'
    if torch.cuda.is_available():
        if cfg.dist.gpu != -1:
            device = torch.device('cuda:{}'.format(cfg.dist.gpu))
        else:
            device = torch.device('cuda')
    else:
        raise ValueError('No GPUs found.')

    train_dataset = hydra.utils.instantiate(cfg.dataloader.dataset.train_dataset)
    test_dataset = hydra.utils.instantiate(cfg.dataloader.dataset.test_dataset)

    if cfg.trainer.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, drop_last=True)
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle=False, drop_last=True)
    else:
        train_sampler = None
        test_sampler = None
    TrainImgLoader = DataLoader(
        train_dataset, cfg.dataloader.batch_size, shuffle=(train_sampler is None),
        num_workers=cfg.dataloader.workers, pin_memory=True, persistent_workers=True, prefetch_factor=4,
        sampler=train_sampler)
    TestImgLoader = DataLoader(
        test_dataset, cfg.dataloader.test_batch_size, shuffle=False,
        num_workers=cfg.dataloader.workers, pin_memory=True, persistent_workers=True, prefetch_factor=4,
        sampler=test_sampler)

    optimizer = hydra.utils.instantiate(cfg.optimizer, params=model.parameters())

    T_max = cfg.trainer.epochs * len(TrainImgLoader)
    if cfg.trainer.lr_scheduler == 'none':
        lr_scheduler = None
    elif cfg.trainer.lr_scheduler == 'onecyc_linear':
        lr_scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, cfg.optimizer.lr, T_max + 100,
            pct_start=0.05,
            cycle_momentum=False, anneal_strategy='linear')
    elif cfg.trainer.lr_scheduler == 'onecyc_cos':
        lr_scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, cfg.optimizer.lr, T_max + 100,
            pct_start=0.05, cycle_momentum=False, anneal_strategy='cos')
    elif cfg.trainer.lr_scheduler == 'cos':
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max, eta_min=1e-8)
    else:
        raise ValueError('Not supported lr scheduler: {}'.format(cfg.trainer.lr_scheduler))

    if utils.is_main_process() and cfg.use_wandb:
        wandb_run_id = wandb.util.generate_id()

    # load parameters
    start_epoch = 0
    if cfg.trainer.loadckpt is not None:
        state_dict = None
        # load the checkpoint file specified by args.loadckpt
        if utils.is_main_process():
            log.info("Loading model {}".format(cfg.trainer.loadckpt))
        if cfg.dist.gpu == -1:
            state_dict = torch.load(cfg.trainer.loadckpt)
        elif torch.cuda.is_available():
            # Map model to be loaded to specified single gpu.
            loc = 'cuda:{}'.format(cfg.dist.gpu)
            state_dict = torch.load(cfg.trainer.loadckpt, map_location=loc)

        state_dict = state_dict['model']
        if list(state_dict.keys())[0].split('.')[1] != 'voxel_net':
            new_state_dict = {}
            for k, v in state_dict.items():
                splits = k.split('.')
                splits.insert(1, 'voxel_net')
                k = '.'.join(splits)
                new_state_dict[k] = v
            state_dict = new_state_dict

        # change strict to False if loading from occupancy checkpoints
        model.load_state_dict(state_dict, strict=True)
    if utils.is_main_process():
        log.info("Start at epoch {}".format(start_epoch))

    if utils.is_main_process() and cfg.use_wandb:
        log.info("Start at epoch {}".format(start_epoch))
        wandb.init(project="odtformer", entity="jerrydty", id=wandb_run_id)
        wandb.run.name = log_info
        wandb.save()

    best_checkpoint_voxel_loss = float('inf')
    for epoch_idx in range(start_epoch, cfg.trainer.epochs):
        # training
        avg_train_scalars = AverageMeterDict()
        avg_train_iou_0 = AverageMeterDict()
        avg_train_iou_1 = AverageMeterDict()
        for batch_idx, sample in enumerate(TrainImgLoader):
            global_step = len(TrainImgLoader) * epoch_idx + batch_idx
            start_time = time.time()
            do_summary = global_step % cfg.trainer.summary_freq == 0
            sum_loss, iou_dict_0, iou_dict_1, scalar_outputs = train_sample(model, sample, optimizer, lr_scheduler, cfg,
                                                                            device)

            if utils.is_main_process():
                if do_summary:
                    last_lr = cfg.optimizer.lr
                    if lr_scheduler is not None:
                        last_lr = lr_scheduler.get_last_lr()[0]

                    log_str = 'Epoch {}/{}, Iter {}/{}, train loss = {:.3f}, IoU_0 = {:.3f}, IoU_1 = {:.3f}, ' \
                              'lr: {:.9f}, time = {:.3f}'.format(epoch_idx,
                                                                 cfg.trainer.epochs,
                                                                 batch_idx,
                                                                 len(TrainImgLoader),
                                                                 sum_loss,
                                                                 iou_dict_0['sum'],
                                                                 iou_dict_1['sum'],
                                                                 last_lr,
                                                                 time.time() - start_time)
                    log.info(log_str)

                    if cfg.use_wandb:
                        wandb.log({'train_IoU_0': iou_dict_0['sum'], 'train_IoU_1': iou_dict_1['sum'],
                                   'train_voxel_loss': scalar_outputs['voxel_loss'],
                                   'train_flow_loss': scalar_outputs['epe_loss']})
                avg_train_scalars.update(scalar_outputs)
                avg_train_iou_0.update(iou_dict_0)
                avg_train_iou_1.update(iou_dict_1)
                del iou_dict_0, iou_dict_1

                # saving checkpoints
                if (epoch_idx + 1) % cfg.trainer.save_freq == 0:
                    checkpoint_data = {'epoch': epoch_idx, 'model': model.state_dict(
                    ), 'optimizer': optimizer.state_dict()}
                    torch.save(checkpoint_data,
                               "{}/{}_checkpoint_{:0>6}.ckpt".format(cfg.trainer.logdir, wandb_run_id, epoch_idx))

        if utils.is_main_process():
            avg_train_scalars = avg_train_scalars.mean()
            avg_train_iou_0 = avg_train_iou_0.mean()
            avg_train_iou_1 = avg_train_iou_1.mean()

            log.info(f"avg_train_voxel_loss: {avg_train_scalars['voxel_loss']}, "
                     f"avg_train_epe_loss: {avg_train_scalars['epe_loss']}")

            if cfg.use_wandb:
                wandb.log({'avg_train_voxel_loss': avg_train_scalars['voxel_loss'],
                           'avg_train_epe_loss': avg_train_scalars['epe_loss'],
                           'weighted_avg_train_IoU_0': avg_train_iou_0['sum'],
                           'weighted_avg_train_IoU_1': avg_train_iou_1['sum'],
                           'train_last_level_IoU_0': avg_train_iou_0['last'],
                           'train_last_level_IoU_1': avg_train_iou_1['last']})

        gc.collect()

        # testing
        avg_test_scalars = AverageMeterDict()
        avg_test_iou_0 = AverageMeterDict()
        avg_test_iou_1 = AverageMeterDict()
        for batch_idx, sample in enumerate(TestImgLoader):
            start_time = time.time()
            iou_dict_0, iou_dict_1, scalar_outputs, voxel_outputs = test_sample(model, sample, cfg, device)
            if utils.is_main_process():
                log.info(
                    'Epoch {}/{}, Iter {}/{}, time = {:.3f}'.format(epoch_idx, cfg.trainer.epochs, batch_idx,
                                                                    len(TestImgLoader), time.time() - start_time))

                avg_test_scalars.update(scalar_outputs)
                avg_test_iou_0.update(iou_dict_0)
                avg_test_iou_1.update(iou_dict_1)
                del scalar_outputs, voxel_outputs, iou_dict_0, iou_dict_1

        if utils.is_main_process():
            avg_test_scalars = avg_test_scalars.mean()
            avg_test_iou_0 = avg_test_iou_0.mean()
            avg_test_iou_1 = avg_test_iou_1.mean()

            log.info(f"avg_test_scalars {avg_test_scalars}")

            if cfg.use_wandb:
                wandb.log({'avg_test_voxel_loss': avg_test_scalars['voxel_loss'],
                           'avg_test_epe_loss': avg_test_scalars['epe_loss'],
                           'weighted_avg_test_IoU_0': avg_test_iou_0['sum'],
                           'weighted_avg_test_IoU_1': avg_test_iou_1['sum'],
                           'test_last_level_IoU_0': avg_test_iou_0['last'],
                           'test_last_level_IoU_1': avg_test_iou_1['last']})

            # saving new best checkpoint
            if avg_test_scalars['voxel_loss'] < best_checkpoint_voxel_loss:
                best_checkpoint_voxel_loss = avg_test_scalars['voxel_loss']
                log.debug("Overwriting best checkpoint")
                checkpoint_data = {'epoch': epoch_idx, 'model': model.state_dict(
                ), 'optimizer': optimizer.state_dict()}
                torch.save(checkpoint_data, "{}/best.ckpt".format(cfg.trainer.logdir))

        gc.collect()


# train one sample
def train_sample(model, sample, optimizer, lr_scheduler, cfg, device):
    model.train()

    imgL_0, imgR_0 = sample['left_0'], sample['right_0']
    imgL_1, imgR_1 = sample['left_1'], sample['right_1']
    voxel_gt_list_0, voxel_gt_list_1 = sample['voxel_grid_0'], sample['voxel_grid_1']
    flow_gt = sample['voxel_flow']
    calib_meta_0 = {'T_world_cam_101': sample['T_world_cam_101_0'], 'T_world_cam_103': sample['T_world_cam_103_0'],
                    'cam_101': sample['cam_101_0'], 'cam_103': sample['cam_103_0']}
    calib_meta_1 = {'T_world_cam_101': sample['T_world_cam_101_1'], 'T_world_cam_103': sample['T_world_cam_103_1'],
                    'cam_101': sample['cam_101_1'], 'cam_103': sample['cam_103_1']}

    if torch.cuda.is_available():
        imgL_0 = imgL_0.to(device, non_blocking=True)
        imgR_0 = imgR_0.to(device, non_blocking=True)
        imgL_1 = imgL_1.to(device, non_blocking=True)
        imgR_1 = imgR_1.to(device, non_blocking=True)
        flow_gt = flow_gt.to(device, non_blocking=True)
        for i in range(len(voxel_gt_list_0)):
            voxel_gt_list_0[i] = voxel_gt_list_0[i].to(device, non_blocking=True)
            voxel_gt_list_1[i] = voxel_gt_list_1[i].to(device, non_blocking=True)

    optimizer.zero_grad()

    pair_0_kwargs = {'calib_meta': calib_meta_0}
    pair_1_kwargs = {'calib_meta': calib_meta_1}
    voxel_ests, flow_est = model((imgL_0, imgR_0), (imgL_1, imgR_1), training=True, pair_0_kwargs=pair_0_kwargs,
                                 pair_1_kwargs=pair_1_kwargs)
    loss_0, _ = model_loss(voxel_ests[0], voxel_gt_list_0, cfg.trainer.loss_weights, cfg.trainer.loss)
    loss_1, _ = model_loss(voxel_ests[1], voxel_gt_list_1, cfg.trainer.loss_weights, cfg.trainer.loss)
    iou_dict_0 = model_iou(voxel_ests[0], voxel_gt_list_0, cfg.trainer.loss_weights)
    iou_dict_1 = model_iou(voxel_ests[1], voxel_gt_list_1, cfg.trainer.loss_weights)

    voxel_loss = (0.5 * loss_0 + 0.5 * loss_1).item()
    epe_loss = scene_epe(flow_est, flow_gt)
    # epe_loss = foreground_epe(flow_est, flow_gt, voxel_gt_list_0[-1])
    loss = 3 * loss_0 + 3 * loss_1 + 4 * epe_loss
    scalar_outputs = {'voxel_loss': voxel_loss, 'epe_loss': epe_loss.item()}

    loss.backward()
    optimizer.step()
    # scaler.scale(loss).backward()
    # scaler.step(optimizer)
    if lr_scheduler is not None:
        lr_scheduler.step()

    # scaler.update()
    return loss.item(), tensor2float(iou_dict_0), tensor2float(iou_dict_1), tensor2float(scalar_outputs)


# test one sample
@make_nograd_func
def test_sample(model, sample, cfg, device):
    model.eval()

    imgL_0, imgR_0 = sample['left_0'], sample['right_0']
    imgL_1, imgR_1 = sample['left_1'], sample['right_1']
    voxel_gt_list_0, voxel_gt_list_1 = sample['voxel_grid_0'], sample['voxel_grid_1']
    flow_gt = sample['voxel_flow']
    calib_meta_0 = {'T_world_cam_101': sample['T_world_cam_101_0'], 'T_world_cam_103': sample['T_world_cam_103_0'],
                    'cam_101': sample['cam_101_0'], 'cam_103': sample['cam_103_0']}
    calib_meta_1 = {'T_world_cam_101': sample['T_world_cam_101_1'], 'T_world_cam_103': sample['T_world_cam_103_1'],
                    'cam_101': sample['cam_101_1'], 'cam_103': sample['cam_103_1']}

    if torch.cuda.is_available():
        imgL_0 = imgL_0.to(device, non_blocking=True)
        imgR_0 = imgR_0.to(device, non_blocking=True)
        imgL_1 = imgL_1.to(device, non_blocking=True)
        imgR_1 = imgR_1.to(device, non_blocking=True)
        flow_gt = flow_gt.to(device, non_blocking=True)
        for i in range(len(voxel_gt_list_0)):
            voxel_gt_list_0[i] = voxel_gt_list_0[i].to(device, non_blocking=True)
            voxel_gt_list_1[i] = voxel_gt_list_1[i].to(device, non_blocking=True)

    pair_0_kwargs = {'calib_meta': calib_meta_0}
    pair_1_kwargs = {'calib_meta': calib_meta_1}
    with torch.no_grad():
        voxel_ests, flow_est = model((imgL_0, imgR_0), (imgL_1, imgR_1), training=False, pair_0_kwargs=pair_0_kwargs,
                                     pair_1_kwargs=pair_1_kwargs)
    loss_0, _ = model_loss(voxel_ests[0], voxel_gt_list_0, cfg.trainer.loss_weights, cfg.trainer.loss)
    loss_1, _ = model_loss(voxel_ests[1], voxel_gt_list_1, cfg.trainer.loss_weights, cfg.trainer.loss)
    iou_dict_0 = model_iou(voxel_ests[0], voxel_gt_list_0, cfg.trainer.loss_weights)
    iou_dict_1 = model_iou(voxel_ests[1], voxel_gt_list_1, cfg.trainer.loss_weights)

    voxel_loss = (0.5 * loss_0 + 0.5 * loss_1).item()
    epe_loss = scene_epe(flow_est, flow_gt)
    # epe_loss = foreground_epe(flow_est, flow_gt, voxel_gt_list_0[-1])
    scalar_outputs = {'voxel_loss': voxel_loss, 'epe_loss': epe_loss.item()}

    # (pair,) level, batch
    voxel_outputs = [voxel_ests[0][1][0], voxel_gt_list_0[1][0]]

    return tensor2float(iou_dict_0), tensor2float(iou_dict_1), tensor2float(scalar_outputs), voxel_outputs


if __name__ == '__main__':
    main()
