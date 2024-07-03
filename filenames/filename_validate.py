import os
from tqdm import tqdm

# change dataset root directory accordingly
root = '/work/vig/Datasets/KITTI_VoxelFlow'
if not os.path.exists(root):
    print('Dataset path invalid')
    exit(1)

# change index filename accordingly
with open('KITTI_raw.txt') as f:
    for line in tqdm(f.readlines(), 'Validating filepaths'):
        files = line.split()
        for _ in files:
            path = os.path.join(root, _)
            if not os.path.exists(path):
                print('Invalid train path:', path)
                exit(1)

    print('filenames all valid')
