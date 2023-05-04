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
import random
import copy
import os
import os.path as osp
import shutil
from glob import glob
from itertools import count
from functools import partial
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from skimage.io import imread, imsave
from tqdm import tqdm


def get_default_parser():
    """
    Get argument parser with commonly used options.
    
    Returns:
        argparse.ArgumentParser: Argument parser with the following arguments:
            --in_dataset_dir: Input dataset directory.
            --out_dataset_dir: Output dataset directory.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--in_dataset_dir',
        type=str,
        required=True,
        help="Input dataset directory.")
    parser.add_argument(
        '--out_dataset_dir', type=str, help="Output dataset directory.")
    return parser


def add_crop_options(parser):
    """
    Add patch cropping related arguments to an argument parser. The parser will be
        modified in place.
    
    Args:
        parser (argparse.ArgumentParser): Argument parser.
    
    Returns:
        argparse.ArgumentParser: Argument parser with the following arguments:
            --crop_size: Size of cropped patches.
            --crop_stride: Stride of sliding windows when cropping patches.
    """

    parser.add_argument(
        '--crop_size', type=int, help="Size of cropped patches.")
    parser.add_argument(
        '--crop_stride',
        type=int,
        help="Stride of sliding windows when cropping patches. `crop_size` will be used only if `crop_size` is not None.",
    )
    return parser


def crop_and_save(path,
                  out_subdir,
                  crop_size,
                  stride,
                  keep_last=False,
                  pad=True,
                  pad_val=0):
    name, ext = osp.splitext(osp.basename(path))
    out_subsubdir = osp.join(out_subdir, name)
    if not osp.exists(out_subsubdir):
        os.makedirs(out_subsubdir)
    img = imread(path)
    h, w = img.shape[:2]
    if h < crop_size or w < crop_size:
        if not pad:
            raise ValueError(
                f"`crop_size` must be smaller than image size. `crop_size` is {crop_size}, but got image size {h}x{w}."
            )
        padded_img = np.full(
            shape=(max(h, crop_size), max(w, crop_size)) + img.shape[2:],
            fill_value=pad_val,
            dtype=img.dtype)
        padded_img[:h, :w] = img
        h, w = padded_img.shape[:2]
        img = padded_img
    counter = count()
    for i in range(0, h, stride):
        i_st = i
        i_ed = i_st + crop_size
        if i_ed > h:
            if keep_last:
                i_st = h - crop_size
                i_ed = h
            else:
                continue
        for j in range(0, w, stride):
            j_st = j
            j_ed = j_st + crop_size
            if j_ed > w:
                if keep_last:
                    j_st = w - crop_size
                    j_ed = w
                else:
                    continue
            imsave(
                osp.join(out_subsubdir, '{}_{}{}'.format(name,
                                                         next(counter), ext)),
                img[i_st:i_ed, j_st:j_ed],
                check_contrast=False)


def crop_patches(crop_size,
                 stride,
                 data_dir,
                 out_dir,
                 subsets=('train', 'val', 'test'),
                 subdirs=('A', 'B', 'label'),
                 glob_pattern='*',
                 max_workers=0,
                 keep_last=False):
    """
    Crop patches from images in specific directories.
    
    Args:
        crop_size (int): Height and width of the cropped patches will be `crop_size`.
        stride (int): Stride of sliding windows when cropping patches.
        data_dir (str): Root directory of the dataset that contains the input images.
        out_dir (str): Directory to save the cropped patches.
        subsets (tuple|list|None, optional): List or tuple of names of subdirectories 
            or None. Images to be cropped should be stored in `data_dir/subset/subdir/` 
            or `data_dir/subdir/` (when `subsets` is set to None), where `subset` is an 
            element of `subsets`. Defaults to ('train', 'val', 'test').
        subdirs (tuple|list, optional): List or tuple of names of subdirectories. Images 
            to be cropped should be stored in `data_dir/subset/subdir/` or 
            `data_dir/subdir/` (when `subsets` is set to None), where `subdir` is an 
            element of `subdirs`. Defaults to ('A', 'B', 'label').
        glob_pattern (str, optional): Glob pattern used to match image files. 
            Defaults to '*', which matches arbitrary file. 
        max_workers (int, optional): Number of worker threads to perform the cropping 
            operation. Deafults to 0.
        keep_last (bool, optional): If True, keep the last patch in each row and each 
            column. The left and upper border of the last patch will be shifted to 
            ensure that size of the patch be `crop_size`. Defaults to False.
    """

    if max_workers < 0:
        raise ValueError("`max_workers` must be a non-negative integer!")

    if subsets is None:
        subsets = ('', )

    print("Cropping patches...")

    if max_workers == 0:
        for subset in subsets:
            for subdir in subdirs:
                paths = glob(
                    osp.join(data_dir, subset, subdir, glob_pattern),
                    recursive=True)
                out_subdir = osp.join(out_dir, subset, subdir)
                for p in tqdm(paths):
                    crop_and_save(
                        p,
                        out_subdir=out_subdir,
                        crop_size=crop_size,
                        stride=stride,
                        keep_last=keep_last)
    else:
        # Concurrently crop image patches
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for subset in subsets:
                for subdir in subdirs:
                    paths = glob(
                        osp.join(data_dir, subset, subdir, glob_pattern),
                        recursive=True)
                    out_subdir = osp.join(out_dir, subset, subdir)
                    for _ in tqdm(
                            executor.map(partial(
                                crop_and_save,
                                out_subdir=out_subdir,
                                crop_size=crop_size,
                                stride=stride),
                                         paths),
                            total=len(paths)):
                        pass


def get_path_tuples(*dirs, glob_pattern='*', data_dir=None):
    """
    Get tuples of image paths. Each tuple corresponds to a sample in the dataset.
    
    Args:
        *dirs (str): Directories that contains the images.
        glob_pattern (str, optional): Glob pattern used to match image files. 
            Defaults to '*', which matches arbitrary file. 
        data_dir (str|None, optional): Root directory of the dataset that 
            contains the images. If not None, `data_dir` will be used to 
            determine relative paths of images. Defaults to None.
    
    Returns:
        list[tuple]: For directories with the following structure:
            ├── img  
            │   ├── im1.png
            │   ├── im2.png
            │   └── im3.png
            │
            ├── mask
            │   ├── im1.png
            │   ├── im2.png
            │   └── im3.png
            └── ...

        `get_path_tuples('img', 'mask', '*.png')` will return list of tuples:
            [('img/im1.png', 'mask/im1.png'), ('img/im2.png', 'mask/im2.png'), ('img/im3.png', 'mask/im3.png')]
    """

    all_paths = []
    for dir_ in dirs:
        paths = glob(osp.join(dir_, glob_pattern), recursive=True)
        paths = sorted(paths)
        if data_dir is not None:
            paths = [osp.relpath(p, data_dir) for p in paths]
        all_paths.append(paths)
    all_paths = list(zip(*all_paths))
    return all_paths


def create_file_list(file_list, path_tuples, sep=' '):
    """
    Create file list.
    
    Args:
        file_list (str): Path of file list to create.
        path_tuples (list[tuple]): See get_path_tuples().
        sep (str, optional): Delimiter to use when writing lines to file list. 
            Defaults to ' '.
    """

    with open(file_list, 'w') as f:
        for tup in path_tuples:
            line = sep.join(tup)
            f.write(line + '\n')


def create_label_list(label_list, labels):
    """
    Create label list.
    
    Args:
        label_list (str): Path of label list to create.
        labels (list[str]|tuple[str]]): Label names.
    """

    with open(label_list, 'w') as f:
        for label in labels:
            f.write(label + '\n')


def link_dataset(src, dst):
    """
    Make a symbolic link to a dataset.
    
    Args:
        src (str): Path of the original dataset.
        dst (str): Path of the symbolic link.
    """

    if osp.exists(dst) and not osp.isdir(dst):
        raise ValueError(f"{dst} exists and is not a directory.")
    elif not osp.exists(dst):
        os.makedirs(dst)
    src = osp.realpath(src)
    name = osp.basename(osp.normpath(src))
    os.symlink(src, osp.join(dst, name), target_is_directory=True)


def copy_dataset(src, dst):
    """
    Make a copy a dataset.
    
    Args:
        src (str): Path of the original dataset.
        dst (str): Path to copy to.
    """

    if osp.exists(dst) and not osp.isdir(dst):
        raise ValueError(f"{dst} exists and is not a directory.")
    elif not osp.exists(dst):
        os.makedirs(dst)

    src = osp.realpath(src)
    name = osp.basename(osp.normpath(src))
    shutil.copytree(src, osp.join(dst, name))


def random_split(samples,
                 ratios=(0.7, 0.2, 0.1),
                 inplace=True,
                 drop_remainder=False):
    """
    Randomly split the dataset into two or three subsets.
    
    Args:
        samples (list): All samples of the dataset.
        ratios (tuple[float], optional): If the length of `ratios` is 2,
            the two elements indicate the ratios of samples used for training
            and evaluation. If the length of `ratios` is 3, the three elements
            indicate the ratios of samples used for training, validation, and 
            testing. Defaults to (0.7, 0.2, 0.1).
        inplace (bool, optional): Whether to shuffle `samples` in place. 
            Defaults to True.
        drop_remainder (bool, optional): Whether to discard the remaining samples.
            If False, the remaining samples will be included in the last subset.
            For example, if `ratios` is (0.7, 0.1) and `drop_remainder` is False, 
            the two subsets after splitting will contain 70% and 30% of the samples, 
            respectively. Defaults to False.
    """

    if not inplace:
        samples = copy.deepcopy(samples)

    if len(samples) == 0:
        raise ValueError("There are no samples!")

    if len(ratios) not in (2, 3):
        raise ValueError("`len(ratios)` must be 2 or 3!")

    random.shuffle(samples)

    n_samples = len(samples)
    acc_r = 0
    st_idx, ed_idx = 0, 0
    splits = []
    for r in ratios:
        acc_r += r
        ed_idx = round(acc_r * n_samples)
        splits.append(samples[st_idx:ed_idx])
        st_idx = ed_idx

    if ed_idx < len(ratios) and not drop_remainder:
        # Append remainder to the last split
        splits[-1].append(splits[ed_idx:])

    return splits
