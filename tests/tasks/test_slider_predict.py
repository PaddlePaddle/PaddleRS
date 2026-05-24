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
import tempfile

import paddlers as pdrs
import paddlers.transforms as T
from testing_utils import CommonTest


class _TestSliderPredictNamespace:
    class TestSliderPredict(CommonTest):
        def test_blocksize_and_overlap_whole(self):
            # Original image size (256, 256)
            with tempfile.TemporaryDirectory() as td:
                # Whole-image inference using predict()
                pred_whole = self.model.predict(self.image_path,
                                                self.transforms)
                pred_whole = pred_whole['label_map']

                # Whole-image inference using slider_predict()
                save_dir = osp.join(td, 'pred1')
                self.model.slider_predict(self.image_path, save_dir, 256, 0,
                                          self.transforms)
                pred1 = T.decode_image(
                    osp.join(save_dir, self.basename),
                    read_raw=True,
                    decode_sar=False)
                self.check_output_equal(pred1.shape, pred_whole.shape)

                # `block_size` == `overlap`
                save_dir = osp.join(td, 'pred2')
                with self.assertRaises(ValueError):
                    self.model.slider_predict(self.image_path, save_dir, 128,
                                              128, self.transforms)

                # `block_size` is a tuple
                save_dir = osp.join(td, 'pred3')
                self.model.slider_predict(self.image_path, save_dir, (128, 32),
                                          0, self.transforms)
                pred3 = T.decode_image(
                    osp.join(save_dir, self.basename),
                    read_raw=True,
                    decode_sar=False)
                self.check_output_equal(pred3.shape, pred_whole.shape)

                # `block_size` and `overlap` are both tuples
                save_dir = osp.join(td, 'pred4')
                self.model.slider_predict(self.image_path, save_dir, (128, 100),
                                          (10, 5), self.transforms)
                pred4 = T.decode_image(
                    osp.join(save_dir, self.basename),
                    read_raw=True,
                    decode_sar=False)
                self.check_output_equal(pred4.shape, pred_whole.shape)

                # `block_size` larger than image size
                save_dir = osp.join(td, 'pred5')
                with self.assertRaises(ValueError):
                    self.model.slider_predict(self.image_path, save_dir, 512, 0,
                                              self.transforms)

        def test_eager_load(self):
            with tempfile.TemporaryDirectory() as td:
                # Lazy
                save_dir = osp.join(td, 'lazy')
                self.model.slider_predict(self.image_path, save_dir, 128, 64,
                                          self.transforms)
                pred_lazy = T.decode_image(
                    osp.join(save_dir, self.basename),
                    read_raw=True,
                    decode_sar=False)

                # Eager
                save_dir = osp.join(td, 'eager')
                self.model.slider_predict(
                    self.image_path,
                    save_dir,
                    128,
                    64,
                    self.transforms,
                    eager_load=True)
                pred_eager = T.decode_image(
                    osp.join(save_dir, self.basename),
                    read_raw=True,
                    decode_sar=False)

                self.check_output_equal(pred_lazy, pred_eager)

        def test_merge_strategy(self):
            with tempfile.TemporaryDirectory() as td:
                # Whole-image inference using predict()
                pred_whole = self.model.predict(self.image_path,
                                                self.transforms)
                pred_whole = pred_whole['label_map']

                # 'keep_first'
                save_dir = osp.join(td, 'keep_first')
                self.model.slider_predict(
                    self.image_path,
                    save_dir,
                    128,
                    64,
                    self.transforms,
                    merge_strategy='keep_first')
                pred_keepfirst = T.decode_image(
                    osp.join(save_dir, self.basename),
                    read_raw=True,
                    decode_sar=False)
                self.check_output_equal(pred_keepfirst.shape, pred_whole.shape)

                # 'keep_last'
                save_dir = osp.join(td, 'keep_last')
                self.model.slider_predict(
                    self.image_path,
                    save_dir,
                    128,
                    64,
                    self.transforms,
                    merge_strategy='keep_last')
                pred_keeplast = T.decode_image(
                    osp.join(save_dir, self.basename),
                    read_raw=True,
                    decode_sar=False)
                self.check_output_equal(pred_keeplast.shape, pred_whole.shape)

                # 'accum'
                save_dir = osp.join(td, 'accum')
                self.model.slider_predict(
                    self.image_path,
                    save_dir,
                    128,
                    64,
                    self.transforms,
                    merge_strategy='accum')
                pred_accum = T.decode_image(
                    osp.join(save_dir, self.basename),
                    read_raw=True,
                    decode_sar=False)
                self.check_output_equal(pred_accum.shape, pred_whole.shape)

                # 'swell'
                save_dir = osp.join(td, 'swell')
                self.model.slider_predict(
                    self.image_path,
                    save_dir,
                    128,
                    64,
                    self.transforms,
                    merge_strategy='swell')
                pred_swell = T.decode_image(
                    osp.join(save_dir, self.basename),
                    read_raw=True,
                    decode_sar=False)
                self.check_output_equal(pred_swell.shape, pred_whole.shape)

        def test_geo_info(self):
            with tempfile.TemporaryDirectory() as td:
                _, geo_info_in = T.decode_image(
                    self.ref_path, read_geo_info=True)
                self.model.slider_predict(self.image_path, td, 128, 0,
                                          self.transforms)
                _, geo_info_out = T.decode_image(
                    osp.join(td, self.basename), read_geo_info=True)
                self.assertEqual(geo_info_out['geo_trans'],
                                 geo_info_in['geo_trans'])
                self.assertEqual(geo_info_out['geo_proj'],
                                 geo_info_in['geo_proj'])

        def test_batch_size(self):
            with tempfile.TemporaryDirectory() as td:
                # batch_size = 1
                save_dir = osp.join(td, 'bs1')
                self.model.slider_predict(
                    self.image_path,
                    save_dir,
                    128,
                    64,
                    self.transforms,
                    merge_strategy='keep_first',
                    batch_size=1)
                pred_bs1 = T.decode_image(
                    osp.join(save_dir, self.basename),
                    read_raw=True,
                    decode_sar=False)

                # batch_size = 4
                save_dir = osp.join(td, 'bs4')
                self.model.slider_predict(
                    self.image_path,
                    save_dir,
                    128,
                    64,
                    self.transforms,
                    merge_strategy='keep_first',
                    batch_size=4)
                pred_bs4 = T.decode_image(
                    osp.join(save_dir, self.basename),
                    read_raw=True,
                    decode_sar=False)
                self.check_output_equal(pred_bs4, pred_bs1)

                # batch_size = 8
                save_dir = osp.join(td, 'bs8')
                self.model.slider_predict(
                    self.image_path,
                    save_dir,
                    128,
                    64,
                    self.transforms,
                    merge_strategy='keep_first',
                    batch_size=8)
                pred_bs8 = T.decode_image(
                    osp.join(save_dir, self.basename),
                    read_raw=True,
                    decode_sar=False)
                self.check_output_equal(pred_bs8, pred_bs1)


class TestSegSliderPredict(_TestSliderPredictNamespace.TestSliderPredict):
    def setUp(self):
        self.model = pdrs.tasks.seg.UNet(in_channels=10)
        self.transforms = T.Compose([T.Normalize([0.5] * 10, [0.5] * 10)])
        self.image_path = "data/ssst/multispectral.tif"
        self.ref_path = self.image_path
        self.basename = osp.basename(self.ref_path)


class TestCDSliderPredict(_TestSliderPredictNamespace.TestSliderPredict):
    def setUp(self):
        self.model = pdrs.tasks.cd.BIT(in_channels=10)
        self.transforms = T.Compose([T.Normalize([0.5] * 10, [0.5] * 10)])
        self.image_path = ("data/ssmt/multispectral_t1.tif",
                           "data/ssmt/multispectral_t2.tif")
        self.ref_path = self.image_path[0]
        self.basename = osp.basename(self.ref_path)
