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

import copy
from PIL import Image

import numpy as np

import paddle
import paddlers.utils.postprocs as P
from testing_utils import CpuCommonTest

__all__ = ['TestPostProgress']


class TestPostProgress(CpuCommonTest):
    def setUp(self):
        self.image = np.asarray(Image.open("data/ssmt/optical_t2.bmp"))
        self.b_label = np.asarray(Image.open("data/ssmt/binary_gt.bmp"))
        self.m_label = np.asarray(Image.open("data/ssmt/multiclass_gt2.bmp"))

    def test_prepro_mask(self):
        mask = copy.deepcopy(self.b_label)
        mask = P.prepro_mask(mask)
        self.assertEqual(len(mask.shape), 2)
        self.assertEqual(mask.dtype, np.uint8)
        self.assertEqual(tuple(np.unique(mask)), (0, 1))
        mask_tensor = paddle.randn((1, 3, 256, 256), dtype="float32")
        mask_tensor = P.prepro_mask(mask_tensor)
        self.assertEqual(len(mask_tensor.shape), 2)
        self.assertEqual(mask_tensor.dtype, np.uint8)
        self.assertEqual(tuple(np.unique(mask_tensor)), (0, 1, 2))

    def test_del_small_connection(self):
        mask = copy.deepcopy(self.b_label)
        mask = P.prepro_mask(mask)
        mask = P.del_small_connection(mask)
        self.assertEqual(mask.shape, self.b_label.shape)
        self.assertEqual(mask.dtype, self.b_label.dtype)
        self.assertEqual(np.unique(mask), np.unique(self.b_label))

    def test_fill_small_holes(self):
        mask = copy.deepcopy(self.b_label)
        mask = P.prepro_mask(mask)
        mask = P.fill_small_holes(mask)
        self.assertEqual(mask.shape, self.b_label.shape)
        self.assertEqual(mask.dtype, self.b_label.dtype)
        self.assertEqual(np.unique(mask), np.unique(self.b_label))

    def test_morphological_operation(self):
        mask = copy.deepcopy(self.b_label)
        mask = P.prepro_mask(mask)
        for op in ["open", "close", "erode", "dilate"]:
            mask = P.morphological_operation(mask, op)
            self.assertEqual(mask.shape, self.b_label.shape)
            self.assertEqual(mask.dtype, self.b_label.dtype)
            self.assertEqual(np.unique(mask), np.unique(self.b_label))

    def test_building_regularization(self):
        mask = copy.deepcopy(self.b_label)
        mask = P.prepro_mask(mask)
        mask = P.building_regularization(mask)
        self.assertEqual(mask.shape, self.b_label.shape)
        self.assertEqual(mask.dtype, self.b_label.dtype)
        self.assertEqual(np.unique(mask), np.unique(self.b_label))

    def test_cut_road_connection(self):
        mask = copy.deepcopy(self.b_label)
        mask = P.prepro_mask(mask)
        mask = P.cut_road_connection(mask)
        self.assertEqual(mask.shape, self.b_label.shape)
        self.assertEqual(mask.dtype, self.b_label.dtype)
        self.assertEqual(np.unique(mask), np.unique(self.b_label))

    def test_conditional_random_field(self):
        if "conditional_random_field" in dir(P):
            mask = copy.deepcopy(self.m_label)
            mask = P.prepro_mask(mask)
            mask = P.conditional_random_field(self.image, mask)
            self.assertEqual(mask.shape, self.m_label.shape)
            self.assertEqual(mask.dtype, self.m_label.dtype)
            self.assertEqual(np.unique(mask), np.unique(self.m_label))

    def test_markov_random_field(self):
        mask = copy.deepcopy(self.m_label)
        mask = P.prepro_mask(mask)
        mask = P.markov_random_field(self.image, mask)
        self.assertEqual(mask.shape, self.m_label.shape)
        self.assertEqual(mask.dtype, self.m_label.dtype)
        self.assertEqual(np.unique(mask), np.unique(self.m_label))

    def test_deal_one_class(self):
        mask = copy.deepcopy(self.m_label)
        mask = P.prepro_mask(mask)
        func = P.morphological_operation
        mask = P.deal_one_class(mask, 1, func, ops="dilate")
        self.assertEqual(mask.shape, self.m_label.shape)
        self.assertEqual(mask.dtype, self.m_label.dtype)
        self.assertEqual(np.unique(mask), np.unique(self.m_label))
