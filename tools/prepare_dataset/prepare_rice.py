#!/usr/bin/env python

import random
import os.path as osp
from glob import iglob
from functools import reduce, partial

from common import (get_default_parser, create_file_list, link_dataset,
                    random_split, get_path_tuples)

SUBSETS = ('train', 'val')
SUBDIRS = ('cloud', 'label')
FILE_LIST_PATTERN = "{subset}.txt"

if __name__ == '__main__':
    parser = get_default_parser()
    parser.add_argument('--seed', type=int, default=None, help="Random seed.")
    parser.add_argument(
        '--ratios',
        type=float,
        nargs='+',
        default=(0.8, 0.2),
        help="Ratios of each subset (train/val or train/val/test).")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    if len(args.ratios) not in (2, 3):
        raise ValueError("Wrong number of ratios!")

    out_dir = osp.join(args.out_dataset_dir,
                       osp.basename(osp.normpath(args.in_dataset_dir)))

    link_dataset(args.in_dataset_dir, args.out_dataset_dir)

    path_tuples = get_path_tuples(
        *(osp.join(out_dir, subdir) for subdir in SUBDIRS),
        glob_pattern='**/*.png',
        data_dir=args.out_dataset_dir)
    splits = random_split(path_tuples, ratios=args.ratios)

    for subset, split in zip(SUBSETS, splits):
        file_list = osp.join(
            args.out_dataset_dir, FILE_LIST_PATTERN.format(subset=subset))
        create_file_list(file_list, split)
        print(f"Write file list to {file_list}.")
