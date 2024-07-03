import logging

import einops
import torch
from .encoding_utils import (
    grid_2d,
    ray_points_snippet,
)

from torch.nn import Module

# logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)
# logger.setLevel(logging.DEBUG)
# logger.setLevel(logging.WARNING)


EPS = 1e-5


def inverse_sigmoid(x, eps=1e-3):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1/x2)


class ImagePositionEncoding(Module):
    def __init__(
        self,
        dim_out: int,
        # [min_x, max_x, min_y, max_y, min_z, max_z]
        ray_points_scale: list = [-2, 2, -1.5, 0, 0.25, 4.25],
        num_samples: int = 64,  # number of ray points
        min_depth: float = 0.25,
        max_depth: float = 5.25,
    ):
        super().__init__()

        self.dim_out = dim_out

        self.ray_points_scale = ray_points_scale
        self.num_samples = num_samples
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.encoder = torch.nn.Sequential(
                torch.nn.Linear(3 * self.num_samples, dim_out),
                torch.nn.ReLU(),
                torch.nn.Linear(dim_out, dim_out),
        )

    def forward(
        self,
        B: int = None,
        T: int = None,
        camera: torch.Tensor = None,
        T_camera_pseudoCam: torch.Tensor = None,
        T_world_pseudoCam: torch.Tensor = None,
        T_local_world: torch.Tensor = None,
    ):
        width, height = camera.size[0, 0]
        width, height = width.round().int().item(), height.round().int().item()
        pos_2d = grid_2d(width, height, output_range=[0.0, width, 0.0, height])
        pos_2d = pos_2d.to(T_camera_pseudoCam.device)
        min_depth = torch.tensor([self.min_depth], device=T_camera_pseudoCam.device)[0]
        max_depth = torch.tensor([self.max_depth], device=T_camera_pseudoCam.device)[0]
        points3d = None

        points3d = ray_points_snippet(
            pos_2d,
            camera,
            T_camera_pseudoCam,
            T_world_pseudoCam,
            T_local_world,
            self.num_samples,
            min_depth,
            max_depth,
        )
        points3d = einops.rearrange(
            points3d,
            "b t h w (n c) -> (b t) h w n c",
            b=B,
            t=T,
            h=height,
            w=width,
            n=self.num_samples,
        )
        points3d[..., 0] = (points3d[..., 0] - self.ray_points_scale[0]) / (
            self.ray_points_scale[1] - self.ray_points_scale[0]
        )
        points3d[..., 1] = (points3d[..., 1] - self.ray_points_scale[2]) / (
            self.ray_points_scale[3] - self.ray_points_scale[2]
        )
        points3d[..., 2] = (points3d[..., 2] - self.ray_points_scale[4]) / (
            self.ray_points_scale[5] - self.ray_points_scale[4]
        )
        points3d = inverse_sigmoid(points3d)
        points3d = einops.rearrange(
            points3d,
            "(b t) h w n c -> (b t) h w (n c)",
            b=B,
            t=T,
            h=height,
            w=width,
            n=self.num_samples,
        )

        encoding = self.encoder(points3d.contiguous())  # B x C x H x W
        encoding = einops.rearrange(
            encoding,
            "(b t) h w c -> b t c h w",
            b=B,
            t=T,
            h=height,
            w=width,
        )
        # logger.debug(f"ray grid encoding {encoding.shape}")
        return encoding
