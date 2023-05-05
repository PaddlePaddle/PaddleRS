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

import os.path as osp
import copy

from .base import BaseDataset
from paddlers.utils import logging, get_encoding, norm_path, is_pic


class ResDataset(BaseDataset):
    """
    Dataset for image restoration tasks.

    Args:
        data_dir (str): Root directory of the dataset.
        file_list (str): Path of the file that contains relative paths of source and target image files.
        transforms (paddlers.transforms.Compose|list): Data preprocessing and data augmentation operators to apply.
        num_workers (int|str, optional): Number of processes used for data loading. If `num_workers` is 'auto',
            the number of workers will be automatically determined according to the number of CPU cores: If 
            there are more than 16 cores, 8 workers will be used. Otherwise, the number of workers will be half 
            the number of CPU cores. Defaults: 'auto'.
        shuffle (bool, optional): Whether to shuffle the samples. Defaults to False.
        sr_factor (int|None, optional): Scaling factor of image super-resolution task. None for other image 
            restoration tasks. Defaults to None.
    """

    _KEYS_TO_KEEP = ['image', 'target']
    _collate_trans_info = True

    def __init__(self,
                 data_dir,
                 file_list,
                 transforms,
                 num_workers='auto',
                 shuffle=False,
                 sr_factor=None,
                 batch_transforms=None):
        super(ResDataset, self).__init__(data_dir, None, transforms,
                                         num_workers, shuffle, batch_transforms)
        self.file_list = list()

        with open(file_list, encoding=get_encoding(file_list)) as f:
            for line in f:
                items = line.strip().split()
                if len(items) > 2:
                    raise ValueError(
                        "A space is defined as the delimiter to separate the source and target image path, " \
                        "so the space cannot be in the source image or target image path, but the line[{}] of " \
                        " file_list[{}] has a space in the two paths.".format(line, file_list))
                items[0] = norm_path(items[0])
                items[1] = norm_path(items[1])
                full_path_im = osp.join(data_dir, items[0])
                full_path_tar = osp.join(data_dir, items[1])
                if not is_pic(full_path_im) or not is_pic(full_path_tar):
                    continue
                if not osp.exists(full_path_im):
                    raise IOError("Source image file {} does not exist!".format(
                        full_path_im))
                if not osp.exists(full_path_tar):
                    raise IOError("Target image file {} does not exist!".format(
                        full_path_tar))
                sample = {
                    'image': full_path_im,
                    'target': full_path_tar,
                }
                if sr_factor is not None:
                    sample['sr_factor'] = sr_factor
                self.file_list.append(sample)
        self.num_samples = len(self.file_list)
        logging.info("{} samples in file {}".format(
            len(self.file_list), file_list))

    def __len__(self):
        return len(self.file_list)
