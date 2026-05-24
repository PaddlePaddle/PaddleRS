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

__all__ = []


class TestClasModel(TestModel):
    DEFAULT_HW = (224, 224)

    def check_output(self, output, target):
        # target is the expected shape of the output
        self.check_output_equal(output.numpy().shape, target)

    def set_inputs(self):
        def _gen_data(specs):
            for spec in specs:
                c = spec.get('in_channels', 3)
                yield [self.get_randn_tensor(c)]

        self.inputs = _gen_data(self.specs)

    def set_targets(self):
        self.targets = [[self.DEFAULT_BATCH_SIZE, spec.get('num_classes', 2)]
                        for spec in self.specs]


class TestCondenseNetV2AModel(TestClasModel):
    MODEL_CLASS = paddlers.rs_models.clas.CondenseNetV2_A

    def set_specs(self):
        self.specs = [
            dict(in_channels=3, num_classes=2),
            dict(in_channels=10, num_classes=2),
            dict(in_channels=3, num_classes=100)
        ]   # yapf: disable


class TestCondenseNetV2BModel(TestClasModel):
    MODEL_CLASS = paddlers.rs_models.clas.CondenseNetV2_B

    def set_specs(self):
        self.specs = [
            dict(in_channels=3, num_classes=2),
            dict(in_channels=10, num_classes=2),
            dict(in_channels=3, num_classes=100)
        ]   # yapf: disable


class TestCondenseNetV2CModel(TestClasModel):
    MODEL_CLASS = paddlers.rs_models.clas.CondenseNetV2_C

    def set_specs(self):
        self.specs = [
            dict(in_channels=3, num_classes=2),
            dict(in_channels=10, num_classes=2),
            dict(in_channels=3, num_classes=100)
        ]   # yapf: disable
