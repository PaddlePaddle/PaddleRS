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

from copy import deepcopy

from paddle.io import Dataset
from paddle.fluid.dataloader.collate import default_collate_fn

from paddlers.utils import get_num_workers


class BaseDataset(Dataset):
    _collate_trans_info = False

    def __init__(self, data_dir, label_list, transforms, num_workers, shuffle):
        super(BaseDataset, self).__init__()

        self.data_dir = data_dir
        self.label_list = label_list
        self.transforms = deepcopy(transforms)
        self.num_workers = get_num_workers(num_workers)
        self.shuffle = shuffle

    def __getitem__(self, idx):
        sample = deepcopy(self.file_list[idx])
        # `trans_info` will be used to store meta info about image shape
        sample['trans_info'] = []
        outputs, trans_info = self.transforms(sample)
        return outputs, trans_info

    def collate_fn(self, batch):
        if self._collate_trans_info:
            return default_collate_fn(
                [s[0] for s in batch]), [s[1] for s in batch]
        else:
            return default_collate_fn([s[0] for s in batch])
