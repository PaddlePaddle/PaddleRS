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

from paddle.io import Dataset
from paddle.io.dataloader.collate import default_collate_fn

from paddlers.utils import get_num_workers
import paddlers.utils.logging as logging
from paddlers.transforms import construct_sample_from_dict, Compose, BatchCompose


class BaseDataset(Dataset):
    _KEYS_TO_KEEP = None
    _KEYS_TO_DISCARD = None
    _collate_trans_info = False

    def __init__(self,
                 data_dir,
                 label_list,
                 transforms,
                 num_workers,
                 shuffle,
                 batch_transforms=None):
        super(BaseDataset, self).__init__()

        self.data_dir = data_dir
        self.label_list = label_list
        self.transforms = copy.deepcopy(transforms)
        if isinstance(self.transforms, list):
            self.transforms = Compose(self.transforms)

        self.num_workers = get_num_workers(num_workers)
        self.shuffle = shuffle
        if isinstance(batch_transforms, list):
            batch_transforms = BatchCompose(batch_transforms)
        self.batch_transforms = batch_transforms

    def __getitem__(self, idx):
        sample = construct_sample_from_dict(self.file_list[idx])
        # `trans_info` will be used to store meta info about image shape
        sample['trans_info'] = []
        sample, trans_info = self.transforms(sample)
        return sample, trans_info

    def collate_fn(self, batch):
        if self._KEYS_TO_KEEP is not None:
            new_batch = []
            for sample, trans_info in batch:
                new_sample = type(sample)()
                for key in self._KEYS_TO_KEEP:
                    if key in sample:
                        new_sample[key] = sample[key]
                new_batch.append((new_sample, trans_info))
            batch = new_batch
        if self._KEYS_TO_DISCARD:
            for key in self._KEYS_TO_DISCARD:
                for s, _ in batch:
                    s.pop(key, None)

        samples = [s[0] for s in batch]

        if self.batch_transforms:
            samples = self.batch_transforms(samples)

        if self._collate_trans_info:
            return default_collate_fn(samples), [s[1] for s in batch]
        else:
            return default_collate_fn(samples)

    def build_collate_fn(self, batch_transforms, collate_fn_constructor=None):
        if self.batch_transforms is not None and batch_transforms:
            logging.warning(
                "The initial `batch_transforms` will be overwritten.")
        if batch_transforms is not None:
            batch_transforms = copy.deepcopy(batch_transforms)
            if isinstance(batch_transforms, list):
                batch_transforms = BatchCompose(batch_transforms)
            self.batch_transforms = batch_transforms
        if collate_fn_constructor:
            self.collate_fn = collate_fn_constructor(self)
