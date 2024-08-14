# ODTFormer: Efficient Obstacle Detection and Tracking with Stereo Cameras Based on Transformer

Official PyTorch codebase for our IROS 2024 Paper! A transformer-based architecture enables efficient detection and 
tracking of voxel occupancies from temporal sequences of stereo pairs.

Note: the `voxel-flow` branch is our entire architecture with the tracking module, to see the occupancy-only 
implementation, please check out the `main` branch.

Tianye Ding*, Hongyu Li*, Huaizu Jiang

Northeastern University, Brown University

[\[Paper\]](https://arxiv.org/abs/2403.14626)
[\[Video\]](https://youtu.be/zyVpXrjTBRI?si=OSZiEf9RoZAVgwMd)
[\[Project Page\]](https://jerrygcding.github.io/odtformer/)

![Architecture](./assets/architecture.png)

## Installation
### Requirements
The code is tested on:
* CentOS Linux 7
* Python 3.10
* PyTorch 2.1.0
* Torchvision 0.16.0
* CUDA 11.8
* GCC 10.1.0

### Create Conda Environment

```bash
conda env create -f environment.yaml
conda activate odtformer
```

## Data Preparation
### SceneFlow Driving
Download and extract SceneFlow Driving RGB images (cleanpass), camera data, disparity, disparity change and optical flow
from the [following link](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html).

We organize the dataset by the following native file tree
```
.
├── camera_data
│       ├── 15mm_focallength
│       │       └── ...
│       └── 35mm_focallength
│               └── ...
├── disparity
│       ├── 15mm_focallength
│       │       └── ...
│       └── 35mm_focallength
│               └── ...
├── disparity_change
│       ├── 15mm_focallength
│       │       └── ...
│       └── 35mm_focallength
│               └── ...
├── flow_gt                     # precomputed ground-truth voxel flow labels (generated)
│       ├── 15mm_focallength
│       │       └── ...
│       └── 35mm_focallength
│               └── ...
├── frames_cleanpass
│       ├── 15mm_focallength
│       │       └── ...
│       └── 35mm_focallength
│               └── ...
├── optical_flow
│       ├── 15mm_focallength
│       │       └── ...
│       └── 35mm_focallength
│               └── ...
└── voxel_gt                    # precomputed ground-truth voxel occupancy labels (generated)
        ├── 15mm_focallength
        │       └── ...
        └── 35mm_focallength
                └── ...
```

### KITTI 2015
Download and extract KITTI 2015 stereo dataset with camera calibration parameters from the 
[following link](https://www.cvlibs.net/datasets/kitti/eval_scene_flow.php).

We organize the dataset by the following file tree
```
.
├── data_scene_flow
│       ├── testing
│       │       └── ...
│       └── training
│               ├── disp_noc_0
│               ├── disp_noc_1
│               ├── disp_occ_0
│               ├── disp_occ_1
│               ├── flow_gt         # precomputed ground-truth voxel flow labels (generated)
│               ├── flow_noc
│               ├── flow_occ
│               ├── image_2
│               ├── image_3
│               ├── voxel_gt_0      # precomputed ground-truth voxel occupancy labels (generated)
│               └── voxel_gt_1      # precomputed ground-truth voxel occupancy labels (generated)
└── data_scene_flow_calib
        ├── testing
        │       └── ...
        └── training
                └── ...
```

Although our dataset implementations support computing ground-truth voxel occupancy and flow labels during runtime, 
empirical experiments show that such practice would consume excessive computation resources and severely impact running 
efficiency.
Thus, we provide an implementations for precomputing and storing ground-truth labels compatible with Slurm array jobs 
within [driving_gt_gen.py](driving_gt_gen.py) and [kitti_gt_gen.py](kitti_gt_gen.py).

To check the validity of the dataset index filepaths, update the directory and file paths accordingly and run 
[filenames/filename_validate.py](filenames/filename_validate.py) as below:
```bash
python filenames/filename_validate.py
```

## Code Structure
**Config files:**
All experiment parameters are specified in config files (as opposed to command-line arguments). See the 
[config/](config/) directory for example config files. 

Note: before launching an experiment, you must update the paths in the config file to point to your own directories, 
especially within following directories and files [config/dataloader/dataset/](config/dataloader/dataset/), 
[config/trainer/](config/trainer/) and [config/eval_model.yaml](config/eval_model.yaml), indicating where to save the 
logs and checkpoints and where to find the training data.

Temporary config changes are also supported through command-line arguments. See section 
[Launching ODTFormer Experiments](#launching-odtformer-experiments) for examples or check out the official documentation 
of [hydra](https://hydra.cc/docs/intro/).
```
.
├── checkpoints                     # directory for storing model checkpoints
├── config                          # directory for config files
│       ├── dataloader              #   dataloader config
│       │       └── dataset         #       dataset config
│       ├── dist                    #   distributed training config
│       ├── model                   #   model config
│       │       ├── backbone        #       image backbone config
│       │       └── decoder_layer   #       transformer decoder layer config
│       ├── optimizer               #   optimizer config
│       └── trainer                 #   trainer config
├── datasets                        # directory for datasets
├── filenames                       # directory for dataset index files
├── models                          # directory for model implementations
└── utils                           # directory for experiment utilities
```

## Launching ODTFormer experiments
### Distributed pretraining on SceneFlow Driving
If you wish to pretrain our tracking module on SceneFlow Driving through distributed data parallel, you can use your own 
trained occupancy checkpoint or download our pretrained 
[checkpoint](https://drive.google.com/file/d/1INJNLer0PDHGf5aUsOjFLHpDtmLaPUMu/view?usp=sharing) and run 
[train_ddp.py](train_ddp.py) as below:
```bash
python train_ddp.py \
use_wandb=True \
model=odtformer model/backbone=enb0fpn \
dataloader=driving_dataloader \
trainer.logdir='./logs_ddp' trainer.logdir_name='odtformer-flow-driving' \
trainer.loadckpt='./checkpoints/DS_occupancy.ckpt' \
dist.port=12554
```
You can specify whether you want to log experiment results onto your own wandb session (also need to change 
corresponding parts within [train_ddp.py](train_ddp.py)), training output log directory, distributed port and any 
additional arguments following the same manner.

### Distributed finetuning on KITTI 2015
To further finetune on KITTI 2015, you can use your own checkpoint from pretraining above or download our pretrained
[checkpoint](https://drive.google.com/file/d/1o2P_sOsWIiqThKo3y_5MighOrCi2FZC3/view?usp=sharing) and run 
[train_ddp.py](train_ddp.py) as below:
```bash
python train_ddp.py \
use_wandb=True \
model=odtformer model/backbone=enb0fpn \
dataloader=kitti_dataloader \
trainer.epochs=50 trainer.logdir='./logs_ddp'  trainer.logdir_name='odtformer-flow-kitti' \
trainer.summary_freq=5 \
trainer.loadckpt='./checkpoints/Driving_flow.ckpt' \
dist.port=12564
```

### Local evaluation
You can use your own trained checkpoint or download our pretrained 
[checkpoint](https://drive.google.com/file/d/1AxcN_ovFBsPuyDR7B_EczSPzzfdqYof3/view?usp=sharing) for evaluation by 
running [test.py](test.py) as below:
```bash
python test.py
```

## Credits
We would like to thank the authors for the following excellent open source projects:
* [StereoVoxelNet](https://github.com/RIVeR-Lab/stereovoxelnet)
* [Deformable-DETR](https://github.com/fundamentalvision/Deformable-DETR)
* [PETR](https://github.com/megvii-research/PETR)
* [CLIP](https://github.com/openai/CLIP)
* [MobileStereoNet](https://github.com/cogsys-tuebingen/mobilestereonet)

## License
See the [LICENSE](LICENSE) file for details about the license under which this code is made available.

## Citation
If you find this repository useful in your research, please consider giving a star :star: and a citation
```bibtex
@article{ding2024odtformer,
  title={ODTFormer: Efficient Obstacle Detection and Tracking with Stereo Cameras Based on Transformer},
  author={Ding, Tianye and Li, Hongyu and Jiang, Huaizu},
  journal={arXiv preprint arXiv:2403.14626},
  year={2024}
}
```