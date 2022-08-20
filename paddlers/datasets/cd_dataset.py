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

import copy
from enum import IntEnum
import os.path as osp

from .base import BaseDataset
from paddlers.utils import logging, get_encoding, norm_path, is_pic


class CDDataset(BaseDataset):
    """
    Dataset for change detection tasks.

    Args:
        data_dir (str): Root directory of the dataset.
        file_list (str): Path of the file that contains relative paths of images and annotation files. When 
            `with_seg_labels` False, each line in the file contains the paths of the bi-temporal images and
            the change mask. When `with_seg_labels` is True, each line in the file contains the paths of the
            bi-temporal images, the path of the change mask, and the paths of the segmentation masks in both
            temporal phases.
        transforms (paddlers.transforms.Compose): Data preprocessing and data augmentation operators to apply.
        label_list (str|None, optional): Path of the file that contains the category names. Defaults to None.
        num_workers (int|str, optional): Number of processes used for data loading. If `num_workers` is 'auto',
            the number of workers will be automatically determined according to the number of CPU cores: If 
            there are more than 16 cores，8 workers will be used. Otherwise, the number of workers will be half 
            the number of CPU cores. Defaults: 'auto'.
        shuffle (bool, optional): Whether to shuffle the samples. Defaults to False.
        with_seg_labels (bool, optional): Set `with_seg_labels` to True if the datasets provides segmentation 
            masks (e.g., building masks in each temporal phase). Defaults to False.
        binarize_labels (bool, optional): Whether to binarize change masks and segmentation masks. 
            Defaults to False.
    """

    def __init__(self,
                 data_dir,
                 file_list,
                 transforms,
                 label_list=None,
                 num_workers='auto',
                 shuffle=False,
                 with_seg_labels=False,
                 binarize_labels=False):
        super(CDDataset, self).__init__(data_dir, label_list, transforms,
                                        num_workers, shuffle)

        DELIMETER = ' '

        # TODO: batch padding
        self.batch_transforms = None
        self.file_list = list()
        self.labels = list()
        self.with_seg_labels = with_seg_labels
        if self.with_seg_labels:
            num_items = 5  # RGB1, RGB2, CD, Seg1, Seg2
        else:
            num_items = 3  # RGB1, RGB2, CD
        self.binarize_labels = binarize_labels

        # TODO: If `label_list` is not None, let the user parse `label_list`.
        if label_list is not None:
            with open(label_list, encoding=get_encoding(label_list)) as f:
                for line in f:
                    item = line.strip()
                    self.labels.append(item)

        with open(file_list, encoding=get_encoding(file_list)) as f:
            for line in f:
                items = line.strip().split(DELIMETER)

                if len(items) != num_items:
                    raise ValueError(
                        "Line[{}] in file_list[{}] has an incorrect number of file paths.".
                        format(line.strip(), file_list))

                items = list(map(norm_path, items))

                full_path_im_t1 = osp.join(data_dir, items[0])
                full_path_im_t2 = osp.join(data_dir, items[1])
                full_path_label = osp.join(data_dir, items[2])
                if not all(
                        map(is_pic, (full_path_im_t1, full_path_im_t2,
                                     full_path_label))):
                    continue
                if not osp.exists(full_path_im_t1):
                    raise IOError("Image file {} does not exist!".format(
                        full_path_im_t1))
                if not osp.exists(full_path_im_t2):
                    raise IOError("Image file {} does not exist!".format(
                        full_path_im_t2))
                if not osp.exists(full_path_label):
                    raise IOError("Label file {} does not exist!".format(
                        full_path_label))

                if with_seg_labels:
                    full_path_seg_label_t1 = osp.join(data_dir, items[3])
                    full_path_seg_label_t2 = osp.join(data_dir, items[4])
                    if not osp.exists(full_path_seg_label_t1):
                        raise IOError("Label file {} does not exist!".format(
                            full_path_seg_label_t1))
                    if not osp.exists(full_path_seg_label_t2):
                        raise IOError("Label file {} does not exist!".format(
                            full_path_seg_label_t2))

                item_dict = dict(
                    image_t1=full_path_im_t1,
                    image_t2=full_path_im_t2,
                    mask=full_path_label)
                if with_seg_labels:
                    item_dict['aux_masks'] = [
                        full_path_seg_label_t1, full_path_seg_label_t2
                    ]

                self.file_list.append(item_dict)

        self.num_samples = len(self.file_list)
        logging.info("{} samples in file {}".format(
            len(self.file_list), file_list))

    def __getitem__(self, idx):
        sample = copy.deepcopy(self.file_list[idx])
        sample = self.transforms.apply_transforms(sample)

        if self.binarize_labels:
            # Requires 'mask' to exist
            sample['mask'] = self._binarize(sample['mask'])
            if 'aux_masks' in sample:
                sample['aux_masks'] = list(
                    map(self._binarize, sample['aux_masks']))

        outputs = self.transforms.arrange_outputs(sample)

        return outputs

    def __len__(self):
        return len(self.file_list)

    def _binarize(self, mask, threshold=127):
        return (mask > threshold).astype('int64')


class MaskType(IntEnum):
    """
    Enumeration of the mask types used in the change detection task.
    """

    CD = 0
    SEG_T1 = 1
    SEG_T2 = 2
