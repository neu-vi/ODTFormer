import logging
from typing import Optional

import einops

import torch

from .image_pos_encoding import (
    ImagePositionEncoding,
)
# from pytorch_lightning.utilities import rank_zero_only

# logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)
# # logger.setLevel(logging.DEBUG)
# logger.setLevel(logging.WARNING)

EPS = 1e-5


class ImageToPatchTokens(torch.nn.Module):
    """
    Patchify an image or image feature (Image grid --> sequence of patches)
    """

    def __init__(self, patch_size: int):
        super(ImageToPatchTokens, self).__init__()
        self.patch_size = patch_size

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        input:
            img:     tensor, (B, T, C, H, W), A snippet of images or image features
        output:
            patches: tensor, (B, T, H/P, W/P, P*P*C), P is the patch size
        """
        patches = einops.rearrange(
            img,
            "b (t dt) c (h dh) (w dw) -> b t h w (dt dh dw c)",
            dh=self.patch_size,
            dw=self.patch_size,
            dt=1,
        )
        return patches


class ImageSeqTokenizer(torch.nn.Module):
    """
    Tokenize an image for Transformer by:
        1. Convert to sequence of patches
        2. Attach ray positional encoding to patches
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        patch_size: int,
        ray_points_scale: list = [-2, 2, -1.5, 0, 0.25, 4.25],
        num_samples: int = 64, 
        min_depth: float = 0.25,
        max_depth: float = 5.25,
    ):
        """
        Args:
            in_channels:  Input number of channels in the image (typically 3)
            out_channels: Output channels required from model
            patch_size:   size of patch to divide image in
            ray_points_scale: [min_x, max_x, min_y, max_y, min_z, max_z] used to normalize the points along each ray
            num_samples:  number of ray points
            min_depth:    minimum depth of the ray points
            max_depth:    maximum depth of the ray points
        """
        super().__init__()
        self.in_channels = in_channels
        # currently unused
        self.out_channels = out_channels
        self.patch_size = patch_size

        self.to_tokens = ImageToPatchTokens(patch_size=self.patch_size)

        patch_encoding_out = self.in_channels

        self.token_position_encoder = ImagePositionEncoding(
            dim_out=patch_encoding_out,
            ray_points_scale=ray_points_scale,
            num_samples=num_samples,
            min_depth=min_depth,
            max_depth=max_depth,
        )

        for m in self.modules():
            if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
                torch.nn.init.xavier_uniform_(m.weight)
                torch.nn.init.constant_(m.bias, 0)

    def forward(
        self,
        images: torch.Tensor,
        camera: Optional[torch.Tensor] = None,
        T_camera_pseudoCam: Optional[torch.Tensor] = None,
        T_world_pseudoCam: Optional[torch.Tensor] = None,
        T_world_local: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        '''
        input:
            images:             (B, T, C, H, W), image features
            camera:             (B, T, 6), camera intrinsics: with, height, fx, fy, cx, cy
            T_camera_pseudoCam: (B, T, 12), pseudo camera to camera transformation
            T_world_pseudoCam:  (B, T, 12), pseudo camera to world transformation
            T_world_local:      (B, 12), world to local transformation
        output:
            ret:                (B, T*H*W, C), tokenized image features with ray position encoding, patch size = 1
        '''
        assert images.dim() == 5, f"Images needs to have 5 dimensions {images.shape}"

        token_pos_enc = self.token_position_encoder(
            B=images.shape[0],
            T=images.shape[1],
            camera=camera,
            T_camera_pseudoCam=T_camera_pseudoCam,
            T_world_pseudoCam=T_world_pseudoCam,
            T_local_world=T_world_local.inverse(),
        )
        
        ret = images + token_pos_enc
        ret = self.to_tokens(ret)
        ret = einops.rearrange(ret, "b t h w c -> b t (h w) c")
        return einops.rearrange(ret, "b t n c -> b (t n) c")

    # @rank_zero_only
    # def log_images(
    #     self,
    #     images: torch.Tensor
    # ):
    #     log_img = {}
    #     # Stack along channel for rotation
    #     T = images.shape[1]
    #     input_snippet = einops.rearrange(images, "b t c h w -> (b t) c h w")
    #     input_snippet = einops.rearrange(
    #         input_snippet, "(b t) c h w -> b c (t h) w", t=T
    #     )
    #     log_img["input"] = input_snippet
    #     return log_img