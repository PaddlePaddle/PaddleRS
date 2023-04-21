#!/usr/bin/env python

# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os.path as osp
from glob import glob

from PIL import Image
from tqdm import tqdm

from common import (get_default_parser, add_crop_options, crop_patches,
                    create_file_list, copy_dataset, create_label_list,
                    get_path_tuples)

# According to the official doc(https://github.com/CAPTAIN-WHU/iSAID_Devkit), 
# the files should be organized as follows:
# 
# iSAID
# ├── test
# │   └── images
# │       ├── P0006.png
# │       ├── ...
# │       └── P0009.png
# ├── train
# │   └── images
# │       ├── P0002_instance_color_RGB.png
# │       ├── P0002_instance_id_RGB.png
# │       ├── P0002.png
# │       ├── ...
# │       ├── P0010_instance_color_RGB.png
# │       ├── P0010_instance_id_RGB.png
# │       └── P0010.png
# └── val
#     └── images
#         ├── P0003_instance_color_RGB.png
#         ├── P0003_instance_id_RGB.png
#         ├── P0003.png
#         ├── ...
#         ├── P0004_instance_color_RGB.png
#         ├── P0004_instance_id_RGB.png
#         └── P0004.png

CLASSES = ('background', 'ship', 'storage_tank', 'baseball_diamond',
           'tennis_court', 'basketball_court', 'ground_track_field', 'bridge',
           'large_vehicle', 'small_vehicle', 'helicopter', 'swimming_pool',
           'roundabout', 'soccer_ball_field', 'plane', 'harbor')
# Refer to https://github.com/Z-Zheng/FarSeg/blob/master/data/isaid.py
COLOR_MAP = [[0, 0, 0], [0, 0, 63], [0, 191, 127], [0, 63, 0], [0, 63, 127],
             [0, 63, 191], [0, 63, 255], [0, 127, 63], [0, 127, 127],
             [0, 0, 127], [0, 0, 191], [0, 0, 255], [0, 63, 63], [0, 127, 191],
             [0, 127, 255], [0, 100, 155]]
SUBSETS = ('train', 'val')
SUBDIR = 'images'
FILE_LIST_PATTERN = "{subset}.txt"
LABEL_LIST_NAME = "labels.txt"
URL = ""


def flatten(nested_list):
    flattened_list = []
    for ele in nested_list:
        if isinstance(ele, list):
            flattened_list.extend(flatten(ele))
        else:
            flattened_list.append(ele)
    return flattened_list


def rgb2mask(rgb):
    palette = flatten(COLOR_MAP)
    # Pad with zero
    palette = palette + [0] * (256 * 3 - len(palette))
    ref = Image.new(mode='P', size=(1, 1))
    ref.putpalette(palette)
    mask = rgb.quantize(palette=ref, dither=0)
    return mask


if __name__ == '__main__':
    parser = get_default_parser()
    parser.add_argument(
        '--crop_size', type=int, help="Size of cropped patches.", default=800)
    parser.add_argument(
        '--crop_stride',
        type=int,
        help="Stride of sliding windows when cropping patches. `crop_size` will be used only if `crop_size` is not None.",
        default=600)
    args = parser.parse_args()

    out_dir = osp.join(args.out_dataset_dir,
                       osp.basename(osp.normpath(args.in_dataset_dir)))

    assert args.crop_size is not None
    # According to https://github.com/CAPTAIN-WHU/iSAID_Devkit/blob/master/preprocess/split.py
    # Set keep_last=True
    crop_patches(
        args.crop_size,
        args.crop_stride,
        data_dir=args.in_dataset_dir,
        out_dir=out_dir,
        subsets=SUBSETS,
        subdirs=(SUBDIR, ),
        glob_pattern='*.png',
        max_workers=8,
        keep_last=True)

    for subset in SUBSETS:
        path_tuples = []
        print(f"Processing {subset} labels...")
        for im_subdir in tqdm(glob(osp.join(out_dir, subset, SUBDIR, "*/"))):
            im_name = osp.basename(im_subdir[:-1])  # Strip trailing '/'
            if '_' in im_name:
                # Do not process labels
                continue
            mask_subdir = osp.join(out_dir, subset, SUBDIR,
                                   im_name + '_instance_color_RGB')
            for mask_path in glob(osp.join(mask_subdir, '*.png')):
                # Convert RGB files to mask files (pseudo color)
                rgb = Image.open(mask_path).convert('RGB')
                mask = rgb2mask(rgb)
                # Write to the original location
                mask.save(mask_path)
            path_tuples.extend(
                get_path_tuples(
                    im_subdir,
                    mask_subdir,
                    glob_pattern='*.png',
                    data_dir=args.out_dataset_dir))
        path_tuples.sort()

        file_list = osp.join(
            args.out_dataset_dir, FILE_LIST_PATTERN.format(subset=subset))
        create_file_list(file_list, path_tuples)
        print(f"Write file list to {file_list}.")

    label_list = osp.join(args.out_dataset_dir, LABEL_LIST_NAME)
    create_label_list(label_list, CLASSES)
    print(f"Write label list to {label_list}.")
