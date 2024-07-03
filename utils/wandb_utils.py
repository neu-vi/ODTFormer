from torch import Tensor
import numpy as np
import matplotlib.pyplot as plt

from models import ref_points_generator


def get_voxel_bbox(vox_grid: Tensor, start, shape, voxel_size, bbox_size=None, color=(0, 255, 0)):
    """

    @param vox_grid:
    @param start: [x, y, z]
    @param shape:
    @param voxel_size:
    @param bbox_size:
    @param color: [r, g, b]
    @return:
    """
    if bbox_size is None:
        bbox_size = voxel_size
    reference_points = ref_points_generator(start, shape, voxel_size, normalize=False).view(-1, 3).numpy()
    vox_grid_np = vox_grid.view(-1).cpu().numpy()

    corner_offset = np.array([bbox_size / 2, -bbox_size / 2])
    corner_map = np.array(np.meshgrid(corner_offset, corner_offset, corner_offset)).T.reshape(-1, 3)

    occupied = np.expand_dims(reference_points[vox_grid_np >= .5], axis=-2)
    corners_l = occupied + corner_map

    output = []
    for corners in corners_l.tolist():
        output.append({'corners': corners, 'color': color})

    return np.array(output)


def get_cmap_cloud(point_cloud: np.ndarray, roi_scale, cmap='viridis'):
    """

    @param point_cloud:
    @param roi_scale: x_min, x_max, y_min, y_max, z_min, z_max
    @param cmap:
    @return:
    """
    assert len(point_cloud.shape) == 2 and point_cloud.shape[-1] == 3

    z_min, z_max = roi_scale[4], roi_scale[5]
    colormap = plt.get_cmap(cmap)
    norm_depth = (point_cloud[:, -1] - z_min) / (z_max - z_min)
    colors_grad = colormap(norm_depth)[:, :-1]  # exclude alpha channel
    colors = (colors_grad * 255).astype(int)

    colored_cloud = np.concatenate([point_cloud, colors], axis=-1)
    start = np.concatenate([np.array([roi_scale[0], roi_scale[2], roi_scale[4]]), np.array([255, 255, 255])], axis=-1)
    end = np.concatenate([np.array([roi_scale[1], roi_scale[3], roi_scale[5]]), np.array([255, 255, 255])], axis=-1)

    return np.concatenate([colored_cloud, start[None, :], end[None, :]], axis=0)
