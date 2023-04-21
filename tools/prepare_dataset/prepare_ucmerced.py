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

import random
import os.path as osp
from glob import iglob
from functools import reduce, partial

from common import (get_default_parser, create_file_list, link_dataset,
                    random_split, create_label_list)

CLASSES = ('agricultural', 'airplane', 'baseballdiamond', 'beach', 'buildings',
           'chaparral', 'denseresidential', 'forest', 'freeway', 'golfcourse',
           'harbor', 'intersection', 'mediumresidential', 'mobilehomepark',
           'overpass', 'parkinglot', 'river', 'runway', 'sparseresidential',
           'storagetanks', 'tenniscourt')
SUBSETS = ('train', 'val', 'test')
SUBDIRS = tuple(osp.join('Images', cls) for cls in CLASSES)
FILE_LIST_PATTERN = "{subset}.txt"
LABEL_LIST_NAME = "labels.txt"
URL = ""

if __name__ == '__main__':
    parser = get_default_parser()
    parser.add_argument('--seed', type=int, default=None, help="Random seed.")
    parser.add_argument(
        '--ratios',
        type=float,
        nargs='+',
        default=(0.7, 0.2, 0.1),
        help="Ratios of each subset (train/val or train/val/test).")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    if len(args.ratios) not in (2, 3):
        raise ValueError("Wrong number of ratios!")

    out_dir = osp.join(args.out_dataset_dir,
                       osp.basename(osp.normpath(args.in_dataset_dir)))

    link_dataset(args.in_dataset_dir, args.out_dataset_dir)

    splits_list = []
    for idx, (cls, subdir) in enumerate(zip(CLASSES, SUBDIRS)):
        pairs = []
        for p in iglob(osp.join(out_dir, subdir, '*.tif')):
            pair = (osp.relpath(p, args.out_dataset_dir), str(idx))
            pairs.append(pair)
        splits = random_split(pairs, ratios=args.ratios)
        splits_list.append(splits)
    splits = map(partial(reduce, list.__add__), zip(*splits_list))

    for subset, split in zip(SUBSETS, splits):
        file_list = osp.join(
            args.out_dataset_dir, FILE_LIST_PATTERN.format(subset=subset))
        create_file_list(file_list, split)
        print(f"Write file list to {file_list}.")

    label_list = osp.join(args.out_dataset_dir, LABEL_LIST_NAME)
    create_label_list(label_list, CLASSES)
    print(f"Write label list to {label_list}.")
