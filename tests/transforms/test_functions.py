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

import numpy as np

import paddlers.transforms as T
from testing_utils import CpuCommonTest
from data import build_input_from_file

__all__ = ['TestMatchHistograms', 'TestMatchByRegression']


def calc_err(a, b):
    a = a.astype('float64')
    b = b.astype('float64')
    return np.abs(a - b).mean()


class TestMatchHistograms(CpuCommonTest):
    def setUp(self):
        self.inputs = [
            build_input_from_file(
                "data/ssmt/test_mixed_binary.txt", prefix="./data/ssmt")
        ]

    def test_output_shape(self):
        decoder = T.DecodeImg()
        for input in copy.deepcopy(self.inputs):
            for sample in input:
                sample = decoder(sample)

                im_out = T.functions.match_histograms(sample['image2'],
                                                      sample['image'])
                self.check_output_equal(im_out.shape, sample['image2'].shape)
                self.assertEqual(im_out.dtype, sample['image2'].dtype)

                im_out = T.functions.match_histograms(sample['image'],
                                                      sample['image2'])
                self.check_output_equal(im_out.shape, sample['image'].shape)
                self.assertEqual(im_out.dtype, sample['image'].dtype)


class TestMatchByRegression(CpuCommonTest):
    def setUp(self):
        self.inputs = [
            build_input_from_file(
                "data/ssmt/test_mixed_binary.txt", prefix="./data/ssmt")
        ]

    def test_output_shape(self):
        decoder = T.DecodeImg()
        for input in copy.deepcopy(self.inputs):
            for sample in input:
                sample = decoder(sample)

                im_out = T.functions.match_by_regression(sample['image2'],
                                                         sample['image'])
                self.check_output_equal(im_out.shape, sample['image2'].shape)
                self.assertEqual(im_out.dtype, sample['image2'].dtype)
                err1 = calc_err(sample['image'], sample['image2'])
                err2 = calc_err(sample['image'], im_out)

                self.assertLessEqual(err2, err1)
                im_out = T.functions.match_by_regression(sample['image'],
                                                         sample['image2'])
                self.check_output_equal(im_out.shape, sample['image'].shape)
                self.assertEqual(im_out.dtype, sample['image'].dtype)
                err1 = calc_err(sample['image'], sample['image2'])
                err2 = calc_err(im_out, sample['image2'])
                self.assertLessEqual(err2, err1)
