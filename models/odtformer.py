import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import hydra

from .image_seq_tokenizer import ImageSeqTokenizer
from .wrappers import Pose, Camera
from .deformable_transformer import DeformableTransformerDecoder

QUERY_SHAPES = [(6, 2, 10)]
VOXEL_SIZES = [3]


def ref_points_generator(start, shape, voxel_size, normalize=True):
    min_x, min_y, min_z = start
    x_range = torch.arange(shape[0], dtype=torch.float) * voxel_size + voxel_size / 2 + min_x
    y_range = torch.arange(shape[1], dtype=torch.float) * voxel_size + voxel_size / 2 + min_y
    z_range = torch.arange(shape[2], dtype=torch.float) * voxel_size + voxel_size / 2 + min_z

    W, H, D = x_range.shape[0], y_range.shape[0], z_range.shape[0]

    grid_x = x_range.view(-1, 1, 1).repeat(1, H, D)
    grid_y = y_range.view(1, -1, 1).repeat(W, 1, D)
    grid_z = z_range.view(1, 1, -1).repeat(W, H, 1)

    coords = torch.stack((grid_x, grid_y, grid_z), 3).float()

    if normalize:
        coords[..., 0] = (coords[..., 0] - torch.min(coords[..., 0])) / (
                torch.max(coords[..., 0]) - torch.min(coords[..., 0]) + 1e-30)
        coords[..., 1] = (coords[..., 1] - torch.min(coords[..., 1])) / (
                torch.max(coords[..., 1]) - torch.min(coords[..., 1]) + 1e-30)
        coords[..., 2] = (coords[..., 2] - torch.min(coords[..., 2])) / (
                torch.max(coords[..., 2]) - torch.min(coords[..., 2]) + 1e-30)

    return coords


def pos2posemb3d(pos, num_pos_feats=128, temperature=10000):
    # https://github.com/megvii-research/PETR/blob/main/projects/mmdet3d_plugin/models/dense_heads/petr_head.py#L29
    scale = 2 * math.pi
    pos = pos * scale
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
    pos_x = pos[..., 0, None] / dim_t
    pos_y = pos[..., 1, None] / dim_t
    pos_z = pos[..., 2, None] / dim_t
    pos_x = torch.stack(
        (pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1
    ).flatten(-2)
    pos_y = torch.stack(
        (pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1
    ).flatten(-2)
    pos_z = torch.stack(
        (pos_z[..., 0::2].sin(), pos_z[..., 1::2].cos()), dim=-1
    ).flatten(-2)
    posemb = torch.cat((pos_y, pos_x, pos_z), dim=-1)
    return posemb


class ODTFormer(nn.Module):
    def __init__(self,
                 backbone,
                 decoder_layer,
                 roi_scale,
                 voxel_sizes,
                 use_ooi_embed,
                 num_decoder_layers=(3, 1),
                 share_decoder=False):
        super(ODTFormer, self).__init__()
        assert len(QUERY_SHAPES) == len(VOXEL_SIZES) == len(num_decoder_layers)

        # 2D feature extractor
        self.backbone = hydra.utils.instantiate(backbone['feature_extractor'])
        self.feat_dim = backbone['feat_dim']
        self.layer = str(backbone['layer'])
        self.num_levels = backbone['num_levels']
        self.level_embeds = nn.Parameter(torch.Tensor(self.num_levels, self.feat_dim))
        self.cam_embeds = nn.Parameter(torch.Tensor(2, self.feat_dim))

        # reconstruction meta
        self.roi_scale = roi_scale  # [-16, 16, -31, 1, 0, 32]  # [min_x, max_x, min_y, max_y, min_z, max_z]
        self.voxel_sizes = voxel_sizes  # [3., 1.5, .75, .375]  # level - 1, 2, 3, 4
        self.depth_num = 64

        self.num_voxels = []
        min_x, max_x, min_y, max_y, min_z, max_z = roi_scale
        for ix in range(len(voxel_sizes)):
            self.num_voxels.append([
                int((max_x - min_x) // voxel_sizes[ix]),
                int((max_y - min_y) // voxel_sizes[ix]),
                int((max_z - min_z) // voxel_sizes[ix]),
            ])

        self.position_encoder = torch.nn.Sequential(
            torch.nn.Linear(self.feat_dim * 3, self.feat_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.feat_dim, self.feat_dim),
        )
        self.share_decoder = share_decoder

        # image tokenizer
        # B, TxHxW, 32
        #######################################################
        self.im_seq_tokenizer = ImageSeqTokenizer(self.feat_dim, -1, 1, self.roi_scale, 64, .05, 32)
        deform_decoder_layer = hydra.utils.instantiate(decoder_layer)
        deform_decoder_l = []
        for num_layers in num_decoder_layers:
            deform_decoder_l.append(DeformableTransformerDecoder(deform_decoder_layer, num_layers, self.feat_dim,
                                                                 self.roi_scale, use_ooi_embed=use_ooi_embed))
        self.deform_decoders = nn.ModuleList(deform_decoder_l)

        # decoder
        ####################################################
        self.deconv2 = nn.Sequential(
            nn.Conv3d(128, 64, kernel_size=2, padding=1, bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, kernel_size=2, bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True))
        self.deconv2_out = nn.Sequential(nn.Conv3d(64, 1, kernel_size=1, bias=False),
                                         nn.Sigmoid())
        self.deconv3 = nn.Sequential(nn.ConvTranspose3d(64, 32, kernel_size=6, stride=2, padding=1, bias=False),
                                     nn.BatchNorm3d(32),
                                     nn.ReLU(inplace=True),
                                     nn.Conv3d(32, 32, kernel_size=3, bias=False),
                                     nn.BatchNorm3d(32),
                                     nn.ReLU(inplace=True))
        self.deconv3_out = nn.Sequential(nn.Conv3d(32, 1, kernel_size=1, bias=False),
                                         nn.Sigmoid())
        self.deconv4 = nn.Sequential(nn.ConvTranspose3d(32, 16, kernel_size=6, stride=2, padding=1, bias=False),
                                     nn.BatchNorm3d(16),
                                     nn.ReLU(inplace=True),
                                     nn.Conv3d(16, 16, kernel_size=3, bias=False),
                                     nn.BatchNorm3d(16),
                                     nn.ReLU(inplace=True))
        self.deconv4_out = nn.Sequential(nn.Conv3d(16, 1, kernel_size=1, bias=False),
                                         nn.Sigmoid())
        self.deconv5 = nn.Sequential(nn.ConvTranspose3d(16, 8, kernel_size=6, stride=2, padding=1, bias=False),
                                     nn.BatchNorm3d(8),
                                     nn.ReLU(inplace=True),
                                     nn.Conv3d(8, 8, kernel_size=3, bias=False),
                                     nn.BatchNorm3d(8),
                                     nn.ReLU(inplace=True))
        self.deconv5_out = nn.Sequential(nn.Conv3d(8, 1, kernel_size=1, bias=False),
                                         nn.Sigmoid())
        ####################################################

        if self.share_decoder:
            self.head = nn.Linear(self.feat_dim, 1, bias=False)
        else:
            head_l = []
            for level_idx in range(len(self.voxel_sizes)):
                head_l.append(nn.Linear(self.feat_dim, 1, bias=False))

            self.head = nn.ModuleList(head_l)
        #######################################################
        # initialization
        self.init_weights()

    def init_weights(self):
        nn.init.normal_(self.level_embeds)
        nn.init.normal_(self.cam_embeds)

    def get_backbone_features(self, im):
        features = self.backbone(im)

        if self.layer != '-1':
            feature_list = []
            for layer in range(4):
                feature = features[str(layer)]
                feature = F.interpolate(
                    feature, features[self.layer].shape[-2:], mode="bilinear"
                )
                feature_list.append(feature)
            f = torch.cat(feature_list, dim=1)
        else:
            f = features

        return f

    def process_occupancy_in_single_level(self, token_seq, level_idx,
                                          *,
                                          parent_feat,
                                          spatial_shapes,
                                          level_start_index,
                                          meta_data_no_cam,
                                          cameras):
        """
        parent_query_feat: B x L x C
        """
        B, _, C = token_seq.shape
        T = cameras[0].shape[1]

        query_shape = QUERY_SHAPES[level_idx]
        voxel_size = VOXEL_SIZES[level_idx]

        #######################################################
        # generate reference points (3D coordinates) for all the voxels
        # X x Y x Z x 3
        reference_points = ref_points_generator(
            start=[self.roi_scale[0], self.roi_scale[2], self.roi_scale[4]],
            shape=query_shape,
            voxel_size=voxel_size
        )
        # B x (X x Y x Z) x 3
        reference_points = reference_points.view(-1, 3).unsqueeze(0).repeat(B, 1, 1).to(token_seq.device)

        if parent_feat is not None:
            offset_feat = self.downsample(parent_feat).view(B, C, -1).permute(0, 2, 1)
            query_feat = self.position_encoder(pos2posemb3d(reference_points, self.feat_dim)) + offset_feat
        else:
            query_feat = self.position_encoder(pos2posemb3d(reference_points, self.feat_dim))

        feature_level = self.num_levels - 1
        meta_data = {'camera': Camera(cameras[feature_level]), **meta_data_no_cam}
        # N, L, T, 6
        camera_l = Camera(torch.stack(cameras, dim=1))

        # num_query, 3 -> N, L, num_query, 3
        ms_reference_points = ref_points_generator(
            start=[self.roi_scale[0], self.roi_scale[2], self.roi_scale[4]],
            shape=query_shape,
            voxel_size=voxel_size,
            normalize=False
        ).view(-1, 3)[None, :, :].repeat(B, 1, 1).to(token_seq.device)

        decoder_out = self.deform_decoders[level_idx](query_feat, token_seq, spatial_shapes,
                                                      level_start_index=level_start_index,
                                                      reference_points=reference_points,
                                                      ms_reference_points=ms_reference_points,
                                                      feature_level=feature_level,
                                                      multiscale_cam=camera_l,
                                                      meta_data=meta_data,
                                                      sample_size=voxel_size).permute(0, 2, 1).contiguous()
        #######################################################
        return decoder_out.view(B, -1, *query_shape)

    def unet_forward_(self, coarse_feat):
        deconv2 = self.deconv2(coarse_feat)
        out_2 = self.deconv2_out(deconv2).squeeze(1)

        deconv3 = self.deconv3(deconv2)
        out_3 = self.deconv3_out(deconv3).squeeze(1)

        deconv4 = self.deconv4(deconv3)
        out_4 = self.deconv4_out(deconv4).squeeze(1)

        deconv5 = self.deconv5(deconv4)
        out_5 = self.deconv5_out(deconv5).squeeze(1)

        return [out_2, out_3, out_4, out_5]

    def forward(self, imL, imR,
                *,
                calib_meta, **kwargs):
        B, iC, iH, iW = imL.shape
        # parse camera parameters
        T_world_cam = torch.stack([calib_meta['T_world_cam_101'], calib_meta['T_world_cam_103']], dim=1)
        # pseudo camera to world transformation
        T_world_pseudoCam = Pose(T_world_cam.to(imL.device)).inverse()
        cam_101 = calib_meta['cam_101']
        cam_103 = calib_meta['cam_103']
        # (B, T, 6), camera intrinsics: with, height, fx, fy, cx, cy
        camera_meta = torch.stack([cam_101, cam_103], dim=-2)
        camera = Camera(camera_meta.to(imL.device))

        featL_dict = self.get_backbone_features(imL)
        featR_dict = self.get_backbone_features(imR)

        spatial_shapes = []
        # format - N, T, L, C, H, W
        seq_l = []
        camera_l = []
        for k in range(self.num_levels):
            _, d_model, lH, lW = featL_dict[str(k)].shape
            spatial_shapes.append([2 * lH, lW])
            seq_l.append(torch.stack([featL_dict[str(k)], featR_dict[str(k)]], dim=1))

            scale = lW / iW, lH / iH
            # N, T, 6
            camera_l.append(camera.scale(scale).data)

        level_start_index = [0]
        for ix, (H_, W_) in enumerate(spatial_shapes):
            level_start_index.append(level_start_index[ix] + H_ * W_)
        level_start_index = torch.tensor(level_start_index[: -1], device=imL.device)

        T = 2
        # identity transformations
        identity = torch.eye(4, dtype=torch.float32).unsqueeze(0).unsqueeze(0).repeat(B, T, 1, 1).to(imL.device)
        # pseudo camera to camera transformation
        T_camera_pseudoCam = Pose.from_4x4mat(identity)
        # world to local transformation
        T_world_local = Pose.from_4x4mat(identity)

        token_seq_l = []
        for _ in range(len(seq_l)):
            # B, (TxHxW), C
            token_seq_ = self.im_seq_tokenizer(seq_l[_], Camera(camera_l[_]), T_camera_pseudoCam, T_world_pseudoCam,
                                               T_world_local).view(B, T, -1, self.feat_dim)
            token_seq_ += self.cam_embeds[None, :, None, :]
            token_seq_l.append(token_seq_.view(B, -1, self.feat_dim) + self.level_embeds[None, None, _, :])

        # B, LxTxLq, C
        token_seq = torch.cat(token_seq_l, dim=-2)

        meta_data_no_cam = {'T_camera_pseudoCam': T_camera_pseudoCam, 'T_world_pseudoCam': T_world_pseudoCam,
                            'T_world_local': T_world_local}

        spatial_shapes = torch.tensor(spatial_shapes, device=token_seq.device)

        # coarse-to-fine occupancy estimation
        parent_feat = None
        if len(kwargs) == 0:
            feat_i = None
            for level_idx in range(len(QUERY_SHAPES)):
                feat_i = self.process_occupancy_in_single_level(
                    token_seq=token_seq,
                    level_idx=level_idx,
                    parent_feat=parent_feat,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    meta_data_no_cam=meta_data_no_cam,
                    cameras=camera_l
                )
                parent_feat = feat_i

            out = self.unet_forward_(feat_i)

            return out
        else:
            raise NotImplementedError
