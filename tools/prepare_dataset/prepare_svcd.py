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

from common import (get_default_parser, get_path_tuples, create_file_list,
                    link_dataset)

SUBSETS = ('train', 'val', 'test')
SUBDIRS = ('A', 'B', 'OUT')
FILE_LIST_PATTERN = "{subset}.txt"
URL = ""

if __name__ == '__main__':
    parser = get_default_parser()
    args = parser.parse_args()

    out_dir = osp.join(args.out_dataset_dir,
                       osp.basename(osp.normpath(args.in_dataset_dir)))

    link_dataset(args.in_dataset_dir, args.out_dataset_dir)

    for subset in SUBSETS:
        # NOTE: Only use cropped real samples.
        path_tuples = get_path_tuples(
            *(osp.join(out_dir, 'Real', 'subset', subset, subdir)
              for subdir in SUBDIRS),
            data_dir=args.out_dataset_dir)
        file_list = osp.join(
            args.out_dataset_dir, FILE_LIST_PATTERN.format(subset=subset))
        create_file_list(file_list, path_tuples)
        print(f"Write file list to {file_list}.")
