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

import paddlers
from rs_models.test_model import TestModel

__all__ = ['TestFarSegModel', 'TestFactSegModel']


class TestSegModel(TestModel):
    DEFAULT_HW = (512, 512)

    def check_output(self, output, target):
        self.assertIsInstance(output, list)
        self.check_output_equal(len(output), len(target))
        for o, t in zip(output, target):
            if isinstance(o, list):
                self.check_output(o, t)
            else:
                o = o.numpy()
                self.check_output_equal(o.shape, t.shape)

    def set_inputs(self):
        def _gen_data(specs):
            for spec in specs:
                c = spec.get('in_channels', 3)
                yield self.get_randn_tensor(c)

        self.inputs = _gen_data(self.specs)

    def set_targets(self):
        def _gen_data(specs):
            for spec in specs:
                c = spec.get('num_classes', 2)
                yield [self.get_zeros_array(c)]

        self.targets = _gen_data(self.specs)


class TestFarSegModel(TestSegModel):
    MODEL_CLASS = paddlers.rs_models.seg.FarSeg

    def set_specs(self):
        base_spec = dict(in_channels=3, num_classes=2)
        self.specs = [
            base_spec,
            dict(in_channels=6, num_classes=10),
            dict(**base_spec,
                backbone='resnet18',
                backbone_pretrained=False),
            dict(**base_spec,
                fpn_out_channels=128,
                fsr_out_channels=64,
                decoder_out_channels=32),
            dict(**base_spec, scale_aware_proj=False)
        ]   # yapf: disable

    def set_targets(self):
        self.targets = [[self.get_zeros_array(2)], [self.get_zeros_array(10)],
                        [self.get_zeros_array(2)], [self.get_zeros_array(2)],
                        [self.get_zeros_array(2)]]


class TestFactSegModel(TestSegModel):
    MODEL_CLASS = paddlers.rs_models.seg.FactSeg

    def set_specs(self):
        base_spec = dict(in_channels=3, num_classes=2)
        self.specs = [
            base_spec,
            dict(in_channels=6, num_classes=10),
            dict(**base_spec,
                 backbone='resnet50',
                 backbone_pretrained=False)
        ]  # yapf: disable

    def set_targets(self):
        self.targets = [[self.get_zeros_array(2)], [self.get_zeros_array(10)],
                        [self.get_zeros_array(2)]]
