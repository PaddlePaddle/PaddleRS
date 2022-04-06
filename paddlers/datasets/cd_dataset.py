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

from paddle.io import Dataset
from paddlers.utils import logging, get_num_workers, get_encoding, path_normalization, is_pic


class CDDataset(Dataset):
    """
    读取变化检测任务数据集，并对样本进行相应的处理（来自SegDataset，图像标签需要两个）。

    Args:
        data_dir (str): 数据集所在的目录路径。
        file_list (str): 描述数据集图片文件和对应标注文件的文件路径（文本内每行路径为相对data_dir的相对路径）。当`with_seg_labels`为
            False（默认设置）时，文件中每一行应依次包含第一时相影像、第二时相影像以及变化检测标签的路径；当`with_seg_labels`为True时，
            文件中每一行应依次包含第一时相影像、第二时相影像、变化检测标签、第一时相建筑物标签以及第二时相建筑物标签的路径。
        label_list (str): 描述数据集包含的类别信息文件路径。默认值为None。
        transforms (paddlers.transforms): 数据集中每个样本的预处理/增强算子。
        num_workers (int|str): 数据集中样本在预处理过程中的线程或进程数。默认为'auto'。
        shuffle (bool): 是否需要对数据集中样本打乱顺序。默认为False。
        with_seg_labels (bool, optional): 数据集中是否包含两个时相的语义分割标签。默认为False。
        binarize_labels (bool, optional): 是否对数据集中的标签进行二值化操作。默认为False。
    """

    def __init__(self,
                 data_dir,
                 file_list,
                 label_list=None,
                 transforms=None,
                 num_workers='auto',
                 shuffle=False,
                 with_seg_labels=False,
                 binarize_labels=False):
        super(CDDataset, self).__init__()

        DELIMETER = ' '

        self.transforms = copy.deepcopy(transforms)
        # TODO: batch padding
        self.batch_transforms = None
        self.num_workers = get_num_workers(num_workers)
        self.shuffle = shuffle
        self.file_list = list()
        self.labels = list()
        self.with_seg_labels = with_seg_labels
        if self.with_seg_labels:
            num_items = 5  # RGB1, RGB2, CD, Seg1, Seg2
        else:
            num_items = 3  # RGB1, RGB2, CD
        self.binarize_labels = binarize_labels

        # TODO：非None时，让用户跳转数据集分析生成label_list
        # 不要在此处分析label file
        if label_list is not None:
            with open(label_list, encoding=get_encoding(label_list)) as f:
                for line in f:
                    item = line.strip()
                    self.labels.append(item)

        with open(file_list, encoding=get_encoding(file_list)) as f:
            for line in f:
                items = line.strip().split(DELIMETER)

                if len(items) != num_items:
                    raise Exception(
                        "Line[{}] in file_list[{}] has an incorrect number of file paths.".
                        format(line.strip(), file_list))

                items = list(map(path_normalization, items))

                full_path_im_t1 = osp.join(data_dir, items[0])
                full_path_im_t2 = osp.join(data_dir, items[1])
                full_path_label = osp.join(data_dir, items[2])
                if not all(
                        map(is_pic, (full_path_im_t1, full_path_im_t2,
                                     full_path_label))):
                    continue
                if not osp.exists(full_path_im_t1):
                    raise IOError('Image file {} does not exist!'.format(
                        full_path_im_t1))
                if not osp.exists(full_path_im_t2):
                    raise IOError('Image file {} does not exist!'.format(
                        full_path_im_t2))
                if not osp.exists(full_path_label):
                    raise IOError('Label file {} does not exist!'.format(
                        full_path_label))

                if with_seg_labels:
                    full_path_seg_label_t1 = osp.join(data_dir, items[3])
                    full_path_seg_label_t2 = osp.join(data_dir, items[4])
                    if not osp.exists(full_path_seg_label_t1):
                        raise IOError('Label file {} does not exist!'.format(
                            full_path_seg_label_t1))
                    if not osp.exists(full_path_seg_label_t2):
                        raise IOError('Label file {} does not exist!'.format(
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
        outputs = self.transforms(sample)
        if self.binarize_labels:
            outputs = outputs[:2] + tuple(map(self._binarize, outputs[2:]))
        return outputs

    def __len__(self):
        return len(self.file_list)

    def _binarize(self, mask, threshold=127):
        return (mask > threshold).astype('int64')


class MaskType(IntEnum):
    """Enumeration of the mask types used in the change detection task."""
    CD = 0
    SEG_T1 = 1
    SEG_T2 = 2
