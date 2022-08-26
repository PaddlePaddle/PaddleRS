#!/usr/bin/env python

# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import argparse
import os
import os.path as osp
import shutil
from glob import glob

from tqdm import tqdm


def get_subdir_name(src_path):
    basename = osp.basename(src_path)
    subdir_name, _ = osp.splitext(basename)
    return subdir_name


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        default='copy',
        type=str,
        choices=['copy', 'link'],
        help="Copy or link images.")
    parser.add_argument(
        "--globs",
        nargs='+',
        type=str,
        help="Glob patterns used to find the images to be copied.")
    parser.add_argument(
        "--tags", nargs='+', type=str, help="Tags of each source directory.")
    parser.add_argument(
        "--save_dir",
        default='./',
        type=str,
        help="Path of directory to save collected results.")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    if len(args.globs) != len(args.tags):
        raise ValueError(
            "The number of globs does not match the number of tags!")

    for pat, tag in zip(args.globs, args.tags):
        im_paths = glob(pat)
        print(f"Glob: {pat}\tTag: {tag}")
        for p in tqdm(im_paths):
            subdir_name = get_subdir_name(p)
            ext = osp.splitext(p)[1]
            subdir_path = osp.join(args.save_dir, subdir_name)
            subdir_path = osp.abspath(osp.normpath(subdir_path))
            if not osp.exists(subdir_path):
                os.makedirs(subdir_path)
            if args.mode == 'copy':
                shutil.copyfile(p, osp.join(subdir_path, tag + ext))
            elif args.mode == 'link':
                os.symlink(p, osp.join(subdir_path, tag + ext))
