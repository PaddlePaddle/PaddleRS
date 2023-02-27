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


class ClasDataset(BaseDataset):
    """
    Dataset for scene classification tasks.

    Args:
        data_dir (str): Root directory of the dataset.
        file_list (str): Path of the file that contains relative paths of images and labels.
        transforms (paddlers.transforms.Compose): Data preprocessing and data augmentation operators to apply.
        label_list (str|None, optional): Path of the file that contains the category names. Defaults to None.
        num_workers (int|str, optional): Number of processes used for data loading. If `num_workers` is 'auto',
            the number of workers will be automatically determined according to the number of CPU cores: If 
            there are more than 16 cores, 8 workers will be used. Otherwise, the number of workers will be half 
            the number of CPU cores. Defaults: 'auto'.
        shuffle (bool, optional): Whether to shuffle the samples. Defaults to False.
    """

    def __init__(self,
                 data_dir,
                 file_list,
                 transforms,
                 label_list=None,
                 num_workers='auto',
                 shuffle=False):
        super(ClasDataset, self).__init__(data_dir, label_list, transforms,
                                          num_workers, shuffle)
        # TODO batch padding
        self.batch_transforms = None
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
                full_path_im = osp.join(data_dir, items[0])
                label = items[1]
                if not is_pic(full_path_im):
                    continue
                if not osp.exists(full_path_im):
                    raise IOError('Image file {} does not exist!'.format(
                        full_path_im))
                if not label.isdigit():
                    raise ValueError(
                        'Label {} does not convert to number(int)!'.format(
                            label))
                self.file_list.append({
                    'image': full_path_im,
                    'label': int(label)
                })
        self.num_samples = len(self.file_list)
        logging.info("{} samples in file {}".format(
            len(self.file_list), file_list))

    def __len__(self):
        return len(self.file_list)
