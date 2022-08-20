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

__all__ = ['TestRCANModel']


class TestResModel(TestModel):
    def check_output(self, output, target):
        pass

    def set_inputs(self):
        pass

    def set_targets(self):
        pass


class TestRCANModel(TestSegModel):
    MODEL_CLASS = paddlers.rs_models.res.RCAN

    def set_specs(self):
        pass
