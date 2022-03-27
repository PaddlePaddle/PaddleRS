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


# 超分辨率数据集定义
class SRdataset(object):
    def __init__(self,
                 mode,
                 gt_floder,
                 lq_floder,
                 transforms,
                 scale,
                 num_workers=4,
                 batch_size=8):
        if mode == 'train':
            preprocess = []
            preprocess.append({
                'name': 'LoadImageFromFile',
                'key': 'lq'
            })  # 加载方式
            preprocess.append({'name': 'LoadImageFromFile', 'key': 'gt'})
            preprocess.append(transforms)  # 变换方式
            self.dataset = {
                'name': 'SRDataset',
                'gt_folder': gt_floder,
                'lq_folder': lq_floder,
                'num_workers': num_workers,
                'batch_size': batch_size,
                'scale': scale,
                'preprocess': preprocess
            }

        if mode == "test":
            preprocess = []
            preprocess.append({'name': 'LoadImageFromFile', 'key': 'lq'})
            preprocess.append({'name': 'LoadImageFromFile', 'key': 'gt'})
            preprocess.append(transforms)
            self.dataset = {
                'name': 'SRDataset',
                'gt_folder': gt_floder,
                'lq_folder': lq_floder,
                'scale': scale,
                'preprocess': preprocess
            }

    def __call__(self):
        return self.dataset


# 对定义的transforms处理方式组合，返回字典
class ComposeTrans(object):
    def __init__(self, input_keys, output_keys, pipelines):
        if not isinstance(pipelines, list):
            raise TypeError(
                'Type of transforms is invalid. Must be List, but received is {}'
                .format(type(pipelines)))
        if len(pipelines) < 1:
            raise ValueError(
                'Length of transforms must not be less than 1, but received is {}'
                .format(len(pipelines)))
        self.transforms = pipelines
        self.output_length = len(output_keys)  # 当output_keys的长度为3时，是DRN训练
        self.input_keys = input_keys
        self.output_keys = output_keys

    def __call__(self):
        pipeline = []
        for op in self.transforms:
            if op['name'] == 'SRPairedRandomCrop':
                op['keys'] = ['image'] * 2
            else:
                op['keys'] = ['image'] * self.output_length
            pipeline.append(op)
        if self.output_length == 2:
            transform_dict = {
                'name': 'Transforms',
                'input_keys': self.input_keys,
                'pipeline': pipeline
            }
        else:
            transform_dict = {
                'name': 'Transforms',
                'input_keys': self.input_keys,
                'output_keys': self.output_keys,
                'pipeline': pipeline
            }

        return transform_dict
