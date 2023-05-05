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

from .base import BaseDataset
from paddlers.utils import logging, get_encoding, norm_path, is_pic


class SegDataset(BaseDataset):
    """
    Dataset for semantic segmentation tasks.

    Args:
        data_dir (str): Root directory of the dataset.
        file_list (str): Path of the file that contains relative paths of images and annotation files.
        transforms (paddlers.transforms.Compose|list): Data preprocessing and data augmentation operators to apply.
        label_list (str|None, optional): Path of the file that contains the category names. Defaults to None.
        num_workers (int|str, optional): Number of processes used for data loading. If `num_workers` is 'auto',
            the number of workers will be automatically determined according to the number of CPU cores: If 
            there are more than 16 cores, 8 workers will be used. Otherwise, the number of workers will be half 
            the number of CPU cores. Defaults: 'auto'.
        shuffle (bool, optional): Whether to shuffle the samples. Defaults to False.
    """

    _KEYS_TO_KEEP = ['image', 'mask']
    _collate_trans_info = True

    def __init__(self,
                 data_dir,
                 file_list,
                 transforms,
                 label_list=None,
                 num_workers='auto',
                 shuffle=False,
                 batch_transforms=None):
        super(SegDataset, self).__init__(data_dir, label_list, transforms,
                                         num_workers, shuffle, batch_transforms)
        self.file_list = list()
        self.labels = list()

        if label_list is not None:
            with open(label_list, encoding=get_encoding(label_list)) as f:
                for line in f:
                    item = line.strip()
                    self.labels.append(item)
        with open(file_list, encoding=get_encoding(file_list)) as f:
            for line in f:
                items = line.strip().split()
                if len(items) > 2:
                    raise ValueError(
                        "A space is defined as the delimiter to separate the image and label path, " \
                        "so the space cannot be in the image or label path, but the line[{}] of " \
                        " file_list[{}] has a space in the image or label path.".format(line, file_list))
                items[0] = norm_path(items[0])
                items[1] = norm_path(items[1])
                full_path_im = osp.join(data_dir, items[0])
                full_path_label = osp.join(data_dir, items[1])
                if not is_pic(full_path_im) or not is_pic(full_path_label):
                    continue
                if not osp.exists(full_path_im):
                    raise IOError('Image file {} does not exist!'.format(
                        full_path_im))
                if not osp.exists(full_path_label):
                    raise IOError('Label file {} does not exist!'.format(
                        full_path_label))
                self.file_list.append({
                    'image': full_path_im,
                    'mask': full_path_label
                })
        self.num_samples = len(self.file_list)
        logging.info("{} samples in file {}".format(
            len(self.file_list), file_list))

    def __len__(self):
        return len(self.file_list)
