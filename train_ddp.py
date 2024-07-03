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
from models import model_loss, model_iou
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
        logdir = os.path.join(cfg.trainer.logdir, cfg.trainer.logdir_name) + '_'
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

    model = hydra.utils.instantiate(cfg.model, _recursive_=False)
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
    all_saved_ckpts = [fn for fn in os.listdir(
        cfg.trainer.logdir) if fn.endswith(".ckpt") and ("best" not in fn)]
    if cfg.trainer.resume and len(all_saved_ckpts) > 0:
        # find all checkpoints file and sort according to epoch id
        all_saved_ckpts = sorted(all_saved_ckpts, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        wandb_run_id = all_saved_ckpts[-1].split('_')[0]
        # use the latest checkpoint file
        loadckpt = os.path.join(cfg.trainer.logdir, all_saved_ckpts[-1])
        if utils.is_main_process():
            log.info("Loading the latest model in logdir: {}".format(loadckpt))
        if cfg.dist.gpu == -1:
            state_dict = torch.load(loadckpt)
        elif torch.cuda.is_available():
            # Map model to be loaded to specified single gpu.
            loc = 'cuda:{}'.format(cfg.dist.gpu)
            state_dict = torch.load(loadckpt, map_location=loc)
        model.load_state_dict(state_dict['model'])
        optimizer.load_state_dict(state_dict['optimizer'])
        start_epoch = state_dict['epoch'] + 1
    elif cfg.trainer.loadckpt:
        # load the checkpoint file specified by args.loadckpt
        if utils.is_main_process():
            log.info("Loading model {}".format(cfg.trainer.loadckpt))
        if cfg.dist.gpu == -1:
            state_dict = torch.load(cfg.trainer.loadckpt)
        elif torch.cuda.is_available():
            # Map model to be loaded to specified single gpu.
            loc = 'cuda:{}'.format(cfg.dist.gpu)
            state_dict = torch.load(cfg.trainer.loadckpt, map_location=loc)
        model.load_state_dict(state_dict['model'])
    if utils.is_main_process():
        log.info("Start at epoch {}".format(start_epoch))

    if utils.is_main_process() and cfg.use_wandb:
        # log inside wandb
        if cfg.trainer.resume:
            wandb.init(project="stereo-voxel-former", entity="stereo-voxel-team", id=wandb_run_id, resume=True)
        else:
            wandb.init(project="stereo-voxel-former", entity="stereo-voxel-team", id=wandb_run_id)

        wandb.run.name = log_info
        wandb.save()

    best_checkpoint_loss = 100
    for epoch_idx in range(start_epoch, cfg.trainer.epochs):

        # training
        avg_train_scalars = AverageMeterDict()
        avg_train_iou = AverageMeterDict()
        for batch_idx, sample in enumerate(TrainImgLoader):
            global_step = len(TrainImgLoader) * epoch_idx + batch_idx
            start_time = time.time()
            do_summary = global_step % cfg.trainer.summary_freq == 0
            loss, scalar_outputs, voxel_outputs, iou_dict = train_sample(
                model, sample, optimizer, lr_scheduler, cfg, device
            )

            if utils.is_main_process():
                if do_summary:
                    save_scalars(logger, 'train', scalar_outputs, global_step)
                    last_lr = cfg.optimizer.lr
                    if lr_scheduler is not None:
                        last_lr = lr_scheduler.get_last_lr()[0]
                    log.info(
                        'Epoch {}/{}, Iter {}/{}, train loss = {:.3f}, IoU = {:.3f}, lr: {:.9f}, time = {:.3f}'.format(
                            epoch_idx,
                            cfg.trainer.epochs,
                            batch_idx,
                            len(TrainImgLoader),
                            loss,
                            iou_dict["sum"],
                            last_lr,
                            time.time() - start_time))
                    if cfg.use_wandb:
                        wandb.log({"train_IoU": iou_dict['sum'], "train_loss": loss})

                avg_train_scalars.update(scalar_outputs)
                avg_train_iou.update(iou_dict)
                del scalar_outputs, voxel_outputs, iou_dict

                # saving checkpoints
                if (epoch_idx + 1) % cfg.trainer.save_freq == 0:
                    checkpoint_data = {'epoch': epoch_idx, 'model': model.state_dict(
                    ), 'optimizer': optimizer.state_dict()}
                    torch.save(
                        checkpoint_data,
                        "{}/{}_checkpoint_{:0>6}.ckpt".format(cfg.trainer.logdir, wandb_run_id, epoch_idx))

        if utils.is_main_process():
            avg_train_scalars = avg_train_scalars.mean()
            avg_train_iou = avg_train_iou.mean()
            log.info(f"avg_train_scalars {avg_train_scalars}")
            if cfg.use_wandb:
                wandb.log({'avg_train_loss': avg_train_scalars['loss'], 'weighted_avg_train_IoU': avg_train_iou['sum'],
                           'train_last_level_IoU': avg_train_iou['last']})

        gc.collect()

        # testing
        avg_test_scalars = AverageMeterDict()
        avg_test_iou = AverageMeterDict()
        for batch_idx, sample in enumerate(TestImgLoader):
            global_step = len(TestImgLoader) * epoch_idx + batch_idx
            start_time = time.time()
            do_summary = global_step % cfg.trainer.summary_freq == 0
            do_vis_log = batch_idx % cfg.trainer.vis_log_freq == 0
            test_loss_tensor, test_iou_tensor, test_loss, scalar_outputs, voxel_outputs, iou_dict = test_sample(
                model, sample, cfg, device
            )
            if utils.is_main_process():
                if do_summary:
                    save_scalars(logger, 'test', scalar_outputs, global_step)
                    log.info(
                        'Epoch {}/{}, Iter {}/{}, test loss = {:.3f}, IoU = {:.3f}, time = {:.3f}'.format(epoch_idx,
                                                                                                          cfg.trainer.epochs,
                                                                                                          batch_idx,
                                                                                                          len(
                                                                                                              TestImgLoader),
                                                                                                          test_loss,
                                                                                                          iou_dict[
                                                                                                              "sum"],
                                                                                                          time.time() - start_time))

                if do_vis_log and cfg.use_wandb:
                    all_cloud_gt = np.frombuffer(sample['filtered_point_cloud'][0], dtype=np.float32).reshape(-1, 3)
                    voxel_est, voxel_gt = voxel_outputs
                    start = [cfg.dataloader.ds_roi[0], cfg.dataloader.ds_roi[2], cfg.dataloader.ds_roi[4]]
                    # change by level
                    voxel_size = cfg.dataloader.ds_vox[1]
                    corners_gt = utils.get_voxel_bbox(voxel_gt, start, [12, 4, 20], voxel_size)
                    corners_est = utils.get_voxel_bbox(voxel_est, start, [12, 4, 20], voxel_size,
                                                       bbox_size=voxel_size / 2, color=[0, 255, 255])
                    #################
                    cloud_gt = utils.get_cmap_cloud(all_cloud_gt, cfg.dataloader.ds_roi)

                    point_scene = wandb.Object3D(
                        {'type': 'lidar/beta', 'boxes': np.concatenate([corners_gt, corners_est], axis=0),
                         'points': cloud_gt})
                    wandb.log({f'test_point_scene / epoch {epoch_idx}': point_scene})

                avg_test_scalars.update(scalar_outputs)
                avg_test_iou.update(iou_dict)
                del scalar_outputs, voxel_outputs, iou_dict

        if utils.is_main_process():
            avg_test_scalars = avg_test_scalars.mean()
            avg_test_iou = avg_test_iou.mean()

            save_scalars(logger, 'fulltest', avg_test_scalars,
                         len(TrainImgLoader) * (epoch_idx + 1))
            log.info(f"avg_test_scalars {avg_test_scalars}")
            if cfg.use_wandb:
                wandb.log({'avg_test_loss': avg_test_scalars['loss'], 'weighted_avg_test_IoU': avg_test_iou['sum'],
                           'test_last_level_IoU': avg_test_iou['last']})

            # saving new best checkpoint
            if avg_test_scalars['loss'] < best_checkpoint_loss:
                best_checkpoint_loss = avg_test_scalars['loss']
                log.debug("Overwriting best checkpoint")
                checkpoint_data = {'epoch': epoch_idx, 'model': model.state_dict(
                ), 'optimizer': optimizer.state_dict()}
                torch.save(checkpoint_data, "{}/best.ckpt".format(cfg.trainer.logdir))

        gc.collect()


# train one sample
def train_sample(model, sample, optimizer, lr_scheduler, cfg, device):
    model.train()

    imgL, imgR, voxel_gt_list = sample['left'], sample['right'], sample['voxel_grid']
    calib_meta = {'T_world_cam_101': sample['T_world_cam_101'], 'T_world_cam_103': sample['T_world_cam_103'],
                  'cam_101': sample['cam_101'], 'cam_103': sample['cam_103'], 'left_top': sample['left_top']}

    if torch.cuda.is_available():
        imgL = imgL.to(device, non_blocking=True)
        imgR = imgR.to(device, non_blocking=True)
        for i in range(len(voxel_gt_list)):
            voxel_gt_list[i] = voxel_gt_list[i].to(device, non_blocking=True)

    optimizer.zero_grad()

    # with torch.cuda.amp.autocast():
    voxel_ests = model(imgL, imgR, calib_meta=calib_meta, training=True, voxel_gt=voxel_gt_list)
    loss, iou = model_loss(voxel_ests, voxel_gt_list, cfg.trainer.loss_weights, cfg.trainer.loss)
    iou_dict = model_iou(voxel_ests, voxel_gt_list, cfg.trainer.loss_weights)

    scalar_outputs = {"loss": loss}
    voxel_outputs = []

    loss.backward()
    optimizer.step()
    # scaler.scale(loss).backward()
    # scaler.step(optimizer)
    if lr_scheduler is not None:
        lr_scheduler.step()

    # scaler.update()
    return tensor2float(loss), tensor2float(scalar_outputs), voxel_outputs, tensor2float(iou_dict)


# test one sample
@make_nograd_func
def test_sample(model, sample, cfg, device):
    model.eval()

    imgL, imgR, voxel_gt = sample['left'], sample['right'], sample['voxel_grid']
    calib_meta = {'T_world_cam_101': sample['T_world_cam_101'], 'T_world_cam_103': sample['T_world_cam_103'],
                  'cam_101': sample['cam_101'], 'cam_103': sample['cam_103'], 'left_top': sample['left_top']}

    if torch.cuda.is_available():
        imgL = imgL.to(device, non_blocking=True)
        imgR = imgR.to(device, non_blocking=True)
        for i in range(len(voxel_gt)):
            voxel_gt[i] = voxel_gt[i].to(device, non_blocking=True)

    voxel_ests = model(imgL, imgR, calib_meta=calib_meta, voxel_gt=None, training=False)

    loss, iou = model_loss(voxel_ests, voxel_gt, cfg.trainer.loss_weights, cfg.trainer.loss)
    iou_dict = model_iou(voxel_ests, voxel_gt, cfg.trainer.loss_weights)

    scalar_outputs = {"loss": loss}
    # level, batch
    voxel_outputs = [voxel_ests[1][0], voxel_gt[1][0]]

    return loss, iou, tensor2float(loss), tensor2float(scalar_outputs), voxel_outputs, tensor2float(iou_dict)


if __name__ == '__main__':
    main()
