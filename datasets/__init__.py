from .ds_dataset import VoxelDSDatasetCalib
from .sf_dataset import VoxelDrivingDataset
from .kitti_dataset import VoxelKITTIDataset, VoxelKITTIRaw

__datasets__ = {
    "voxelkitti": VoxelKITTIDataset,
    'voxeldriving': VoxelDrivingDataset,
    "voxeldscalib": VoxelDSDatasetCalib,
}
