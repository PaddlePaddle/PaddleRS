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

from itertools import cycle

from rs_models.test_model import TestModel

__all__ = []


class TestDetModel(TestModel):
    DEFAULT_HW = (608, 608)

    def check_output(self, output, target):
        self.assertIsInstance(output, dict)
        self.assertIsInstance(output['mask'], list)
        self.assertIn('bbox', output)
        self.assertIn('bbox_num', output)
        if 'mask' in output:
            self.assertIsInstance(output['mask'], list)

    def set_inputs(self):
        self.inputs = cycle([self.get_randn_tensor(3)])

    def set_targets(self):
        self.targets = cycle([None])
