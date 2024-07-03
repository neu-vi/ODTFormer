import re
import numpy as np
import torchvision.transforms as transforms


def get_transform(color_jitter):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transform_list = []
    if color_jitter:
        transform_list.append(transforms.ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.4,
        ))
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize(mean=mean, std=std))
    return transforms.Compose(transform_list)


def read_all_lines(filename):
    with open(filename) as f:
        lines = [line.rstrip() for line in f.readlines()]
    return lines


def pfm_imread(filename):
    file = open(filename, 'rb')
    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().decode('utf-8').rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  
        endian = '<'
        scale = -scale
    else:
        endian = '>'  

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data, scale
