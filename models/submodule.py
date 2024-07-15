"""
Modified for [ODTFormer] by Tianye Ding
Original header:
---------------------------------------------------------------------------
Copyright 2021 Faranak Shamsafar

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
---------------------------------------------------------------------------
"""

from __future__ import print_function
import torch
import torch.utils.data


###############################################################################
""" Loss Function """


###############################################################################

def calc_IoU(pred, gt):
    intersect = pred * gt
    total = pred + gt
    union = total - intersect

    return (intersect.sum() + 1.0) / (union.sum() + 1.0)


def IoU_loss(pred, gt):
    return 1 - calc_IoU(pred, gt)


def BCE_loss(pred, gt):
    return torch.nn.functional.binary_cross_entropy(pred, gt.float())


def model_loss(voxel_ests, voxel_gt, weight, loss_type):
    assert loss_type in ['bce', 'iou']
    # voxel_ests = voxel_ests[0]

    # sum of loss of every level
    # from OGN https://openaccess.thecvf.com/content_ICCV_2017/papers/Tatarchenko_Octree_Generating_Networks_ICCV_2017_paper.pdf
    all_losses = []
    all_ious = []

    if isinstance(voxel_ests, torch.Tensor):
        # straight network
        est_shape = voxel_ests.shape
        for idx in range(len(voxel_gt) - 1, -1, -1):
            gt_shape = voxel_gt[idx].shape
            if est_shape[1] == gt_shape[1] and est_shape[2] == gt_shape[2] and est_shape[3] == gt_shape[3]:
                if loss_type == 'bce':
                    all_losses.append(BCE_loss(voxel_ests, voxel_gt[idx]))
                elif loss_type == 'iou':
                    all_losses.append(IoU_loss(voxel_ests, voxel_gt[idx]))
                all_ious.append(calc_IoU(voxel_ests, voxel_gt[idx]))
        assert len(all_losses) == 1, len(all_losses)
        return sum(all_losses), sum(all_ious)

    if isinstance(voxel_ests[0], dict):
        for idx, est_dict in enumerate(voxel_ests):
            assert 'voxel_est' and 'topk_idxes' in est_dict

            topk_idxes = est_dict['topk_idxes']
            voxel_est = est_dict['voxel_est']
            B = voxel_est.shape[0]
            voxel_gt_ = voxel_gt[idx].view(B, -1).gather(1, topk_idxes)
            if loss_type == 'bce':
                all_losses.append(weight[idx] * BCE_loss(voxel_est, voxel_gt_))
            elif loss_type == 'iou':
                all_losses.append(weight[idx] * IoU_loss(voxel_est, voxel_gt_))
        return sum(all_losses), 1 - sum(all_losses)

    if isinstance(voxel_ests[0], torch.Tensor):
        for idx, voxel_est in enumerate(voxel_ests):
            if loss_type == 'bce':
                all_losses.append(weight[idx] * BCE_loss(voxel_est, voxel_gt[idx]))
            elif loss_type == 'iou':
                all_losses.append(weight[idx] * IoU_loss(voxel_est, voxel_gt[idx]))
        return sum(all_losses), 1 - sum(all_losses)

    raise NotImplementedError


def model_iou(voxel_ests, voxel_gt, weight):
    all_ious = []
    last_iou = None
    if isinstance(voxel_ests, torch.Tensor):
        # straight network
        est_shape = voxel_ests.shape
        for idx in range(len(voxel_gt) - 1, -1, -1):
            gt_shape = voxel_gt[idx].shape
            if est_shape[1] == gt_shape[1] and est_shape[2] == gt_shape[2] and est_shape[3] == gt_shape[3]:
                iou = calc_IoU(voxel_ests, voxel_gt[idx])
                all_ious.append(iou)
                last_iou = iou
        assert len(all_ious) == 1, len(all_ious)

    elif isinstance(voxel_ests[0], dict):
        for idx, est_dict in enumerate(voxel_ests):
            assert 'voxel_est' and 'topk_idxes' in est_dict

            topk_idxes = est_dict['topk_idxes']
            voxel_est = est_dict['voxel_est']
            B = voxel_est.shape[0]
            voxel_gt_ = voxel_gt[idx].view(B, -1).gather(1, topk_idxes)
            iou = calc_IoU(voxel_est, voxel_gt_)
            all_ious.append(weight[idx] * iou)
            last_iou = iou

    elif isinstance(voxel_ests[0], torch.Tensor):
        for idx, voxel_est in enumerate(voxel_ests):
            iou = calc_IoU(voxel_est, voxel_gt[idx])
            all_ious.append(weight[idx] * iou)
            last_iou = iou

    else:
        raise NotImplementedError

    return {'sum': sum(all_ious), 'last': last_iou}


def eval_metric(voxel_ests, voxel_gt, metric_func, *args, depth_range=None):
    """

    @param voxel_ests:
    @param voxel_gt:
    @param metric_func:
    @param depth_range:
    @return: Dict{%{depth_range[0]}: [lv0, lv1, lv2, ...], %{depth_range[1]}: [...], ...}
    """
    if depth_range is None:
        depth_range = [1.]

    out_dict = {}
    for r in depth_range:
        out_dict[str(r)] = []

    if isinstance(voxel_ests, torch.Tensor):
        est_shape = voxel_ests.shape
        for idx in range(len(voxel_gt) - 1, -1, -1):
            gt_shape = voxel_gt[idx].shape
            if est_shape[1] == gt_shape[1] and est_shape[2] == gt_shape[2] and est_shape[3] == gt_shape[3]:
                for depth_r in depth_range:
                    z = int(est_shape[-1] * depth_r)
                    if len(args) == 0:
                        metric = metric_func(voxel_ests[..., :z], voxel_gt[idx][..., :z])
                    else:
                        metric = metric_func(voxel_ests[..., :z], voxel_gt[idx][..., :z], args[0][idx])
                    out_dict[str(depth_r)].append(metric)

    elif isinstance(voxel_ests[0], torch.Tensor):
        for idx, voxel_est in enumerate(voxel_ests):
            for depth_r in depth_range:
                z = int(voxel_est.shape[-1] * depth_r)
                if len(args) == 0:
                    metric = metric_func(voxel_est[..., :z], voxel_gt[idx][..., :z])
                else:
                    metric = metric_func(voxel_est[..., :z], voxel_gt[idx][..., :z], args[0][idx])
                out_dict[str(depth_r)].append(metric)

    else:
        raise NotImplementedError

    return out_dict
