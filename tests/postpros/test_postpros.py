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
        self.image1 = np.asarray(Image.open("data/ssmt/optical_t2.bmp"))
        self.image2 = np.asarray(Image.open("data/ssmt/optical_t2.bmp"))
        self.b_label = np.asarray(Image.open("data/ssmt/binary_gt.bmp")).clip(0,
                                                                              1)
        self.m_label = np.asarray(Image.open("data/ssmt/multiclass_gt2.png"))

    def test_prepro_mask(self):
        mask = copy.deepcopy(self.b_label)
        mask = P.prepro_mask(mask)
        self.check_output_equal(len(mask.shape), 2)
        self.assertEqual(mask.dtype, np.uint8)
        self.check_output_equal(np.unique(mask), np.array([0, 1]))
        mask_tensor = paddle.randn((1, 3, 256, 256), dtype="float32")
        mask_tensor = P.prepro_mask(mask_tensor)
        self.check_output_equal(len(mask_tensor.shape), 2)
        self.assertEqual(mask_tensor.dtype, np.uint8)
        self.check_output_equal(np.unique(mask_tensor), np.array([0, 1, 2]))

    def test_del_small_connection(self):
        mask = copy.deepcopy(self.b_label)
        mask = P.prepro_mask(mask)
        mask = P.del_small_connection(mask)
        self.check_output_equal(mask.shape, self.b_label.shape)
        self.assertEqual(mask.dtype, self.b_label.dtype)
        self.check_output_equal(np.unique(mask), np.unique(self.b_label))

    def test_fill_small_holes(self):
        mask = copy.deepcopy(self.b_label)
        mask = P.prepro_mask(mask)
        mask = P.fill_small_holes(mask)
        self.check_output_equal(mask.shape, self.b_label.shape)
        self.assertEqual(mask.dtype, self.b_label.dtype)
        self.check_output_equal(np.unique(mask), np.unique(self.b_label))

    def test_morphological_operation(self):
        mask = copy.deepcopy(self.b_label)
        mask = P.prepro_mask(mask)
        for op in ["open", "close", "erode", "dilate"]:
            mask = P.morphological_operation(mask, op)
            self.check_output_equal(mask.shape, self.b_label.shape)
            self.assertEqual(mask.dtype, self.b_label.dtype)
            self.check_output_equal(np.unique(mask), np.unique(self.b_label))

    def test_building_regularization(self):
        mask = copy.deepcopy(self.b_label)
        mask = P.prepro_mask(mask)
        mask = P.building_regularization(mask)
        self.check_output_equal(mask.shape, self.b_label.shape)
        self.assertEqual(mask.dtype, self.b_label.dtype)
        self.check_output_equal(np.unique(mask), np.unique(self.b_label))

    def test_cut_road_connection(self):
        mask = copy.deepcopy(self.b_label)
        mask = P.prepro_mask(mask)
        mask = P.cut_road_connection(mask)
        self.check_output_equal(mask.shape, self.b_label.shape)
        self.assertEqual(mask.dtype, self.b_label.dtype)
        self.check_output_equal(np.unique(mask), np.unique(self.b_label))

    def test_conditional_random_field(self):
        if "conditional_random_field" in dir(P):
            mask = copy.deepcopy(self.m_label)
            mask = P.prepro_mask(mask)
            mask = P.conditional_random_field(self.image2, mask)
            self.check_output_equal(mask.shape, self.m_label.shape)
            self.assertEqual(mask.dtype, self.m_label.dtype)
            self.check_output_equal(np.unique(mask), np.unique(self.m_label))

    def test_markov_random_field(self):
        mask = copy.deepcopy(self.m_label)
        mask = P.prepro_mask(mask)
        mask = P.markov_random_field(self.image2, mask)
        self.check_output_equal(mask.shape, self.m_label.shape)
        self.assertEqual(mask.dtype, self.m_label.dtype)
        self.check_output_equal(np.unique(mask), np.unique(self.m_label))

    def test_deal_one_class(self):
        mask = copy.deepcopy(self.m_label)
        mask = P.prepro_mask(mask)
        func = P.morphological_operation
        mask = P.deal_one_class(mask, 1, func, ops="dilate")
        self.check_output_equal(mask.shape, self.m_label.shape)
        self.assertEqual(mask.dtype, self.m_label.dtype)
        self.check_output_equal(np.unique(mask), np.unique(self.m_label))

    def test_change_(self):
        mask = copy.deepcopy(self.m_label)
        mask = P.prepro_mask(mask)
        mask = P.change_detection_filter(mask, self.image1, self.image2, 0.8,
                                         0.8, "GLI", {"b": 3,
                                                      "g": 2,
                                                      "r": 1})
        self.check_output_equal(mask.shape, self.m_label.shape)
        self.assertEqual(mask.dtype, self.m_label.dtype)
        self.check_output_equal(np.unique(mask), np.unique(self.m_label))
