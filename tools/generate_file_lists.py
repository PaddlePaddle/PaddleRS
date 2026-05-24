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

import argparse
import os.path as osp

from prepare_dataset.common import get_path_tuples, create_file_list


def gen_file_lists(
        data_dir,
        save_dir,
        subsets=None,
        subdirs=('images', 'masks'),
        glob_pattern='*',
        file_list_pattern="{subset}.txt",
        store_abs_path=False,
        sep=' ', ):
    """
    Generate file lists.

    Args:
        data_dir (str): Root directory of the dataset.
        save_dir (str): Directory to save the generated file lists.
        subsets (tuple|list|None, optional): List or tuple of names of subsets or None. 
            Images should be stored in `data_dir/subset/subdir/` or `data_dir/subdir/` 
            (when `subsets` is set to None), where `subset` is an element of `subsets`. 
            Defaults to None.
        subdirs (tuple|list, optional): List or tuple of names of subdirectories. Images
            should be stored in `data_dir/subset/subdir/` or `data_dir/subdir/` (when 
            `subsets` is set to None), where `subdir` is an element of `subdirs`. 
            Defaults to ('images', 'masks').
        glob_pattern (str, optional): Glob pattern used to match image files. Defaults 
            to '*', which matches arbitrary file.
        file_list_pattern (str, optional): Pattern to name the file lists. Defaults to 
            '{subset}.txt'.
        store_abs_path (bool, optional):  Whether to store the absolute path in file 
            lists. Defaults to 'False', which indicates storing the relative path.
        sep (str, optional): Delimiter to use when writing lines to file lists.
            Defaults to ' '.
    """
    if subsets is None:
        subsets = ('', )
    for subset in subsets:
        path_tuples = get_path_tuples(
            *(osp.join(data_dir, subset, subdir) for subdir in subdirs),
            glob_pattern=glob_pattern,
            data_dir=data_dir)
        if store_abs_path:
            path_tuples_new = []
            for path_tuple in path_tuples:
                path_tuple_new = [
                    osp.abspath(osp.join(data_dir, path_t))
                    for path_t in path_tuple
                ]
                path_tuples_new.append(tuple(path_tuple_new))
            path_tuples = path_tuples_new

        if len(subset) > 0:
            file_list_name = file_list_pattern.format(subset=subset)
        else:
            file_list_name = 'list.txt'
        file_list = osp.join(save_dir, file_list_name)
        create_file_list(file_list, path_tuples, sep)
        print(f"File list {file_list} created.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_dir', type=str, help="Root directory of the dataset.")
    parser.add_argument(
        '--save_dir',
        type=str,
        default='./',
        help="Directory to save the generated file lists.")
    parser.add_argument(
        '--subsets',
        nargs="*",
        default=None,
        help="List or tuple of names of subsets.", )
    parser.add_argument(
        '--subdirs',
        nargs="*",
        default=['A', 'B', 'label'],
        help="List or tuple of names of subdirectories of subsets.", )
    parser.add_argument(
        '--glob_pattern',
        type=str,
        default='*',
        help="Glob pattern used to match image files.", )
    parser.add_argument(
        '--file_list_pattern',
        type=str,
        default='{subset}.txt',
        help="Pattern to name the file lists.", )
    parser.add_argument(
        '--store_abs_path',
        action='store_true',
        help='Whether to store the absolute path in file lists.', )
    parser.add_argument(
        '--sep',
        type=str,
        default=' ',
        help="Delimiter to use when writing lines to file lists.", )
    args = parser.parse_args()

    gen_file_lists(
        data_dir=args.data_dir,
        save_dir=args.save_dir,
        subsets=args.subsets,
        subdirs=args.subdirs,
        glob_pattern=args.glob_pattern,
        file_list_pattern=args.file_list_pattern,
        store_abs_path=args.store_abs_path,
        sep=args.sep, )
