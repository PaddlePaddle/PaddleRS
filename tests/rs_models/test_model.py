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

import inspect

import paddle
import numpy as np
from paddle.static import InputSpec

from paddlers.utils import logging
from testing_utils import CommonTest


class _TestModelNamespace:
    class TestModel(CommonTest):
        MODEL_CLASS = None
        DEFAULT_HW = (256, 256)
        DEFAULT_BATCH_SIZE = 2

        def setUp(self):
            self.set_specs()
            self.set_inputs()
            self.set_targets()
            self.set_models()

        def test_forward(self):
            for i, (
                    input, model, target
            ) in enumerate(zip(self.inputs, self.models, self.targets)):
                try:
                    if isinstance(input, list):
                        output = model(*input)
                    else:
                        output = model(input)
                    self.check_output(output, target)
                except:
                    logging.warning(f"Model built with spec{i} failed!")
                    raise

        def test_to_static(self):
            for i, (
                    input, model, target
            ) in enumerate(zip(self.inputs, self.models, self.targets)):
                try:
                    static_model = paddle.jit.to_static(
                        model, input_spec=self.get_input_spec(model, input))
                except:
                    logging.warning(f"Model built with spec{i} failed!")
                    raise

        def check_output(self, output, target):
            pass

        def set_specs(self):
            self.specs = []

        def set_models(self):
            self.models = (self.build_model(spec) for spec in self.specs)

        def set_inputs(self):
            self.inputs = []

        def set_targets(self):
            self.targets = []

        def build_model(self, spec):
            if '_phase' in spec:
                phase = spec.pop('_phase')
            else:
                phase = 'train'
            if '_stop_grad' in spec:
                stop_grad = spec.pop('_stop_grad')
            else:
                stop_grad = False

            model = self.MODEL_CLASS(**spec)

            if phase == 'train':
                model.train()
            elif phase == 'eval':
                model.eval()
                if stop_grad:
                    for p in model.parameters():
                        p.stop_gradient = True

            return model

        def get_shape(self, c, b=None, h=None, w=None):
            if h is None or w is None:
                h, w = self.DEFAULT_HW
            if b is None:
                b = self.DEFAULT_BATCH_SIZE
            return (b, c, h, w)

        def get_zeros_array(self, c, b=None, h=None, w=None):
            shape = self.get_shape(c, b, h, w)
            return np.zeros(shape)

        def get_randn_tensor(self, c, b=None, h=None, w=None):
            shape = self.get_shape(c, b, h, w)
            return paddle.randn(shape)

        def get_input_spec(self, model, input):
            if not isinstance(input, list):
                input = [input]
            input_spec = []
            for param_name, tensor in zip(
                    inspect.signature(model.forward).parameters, input):
                # XXX: Hard-code dtype
                input_spec.append(
                    InputSpec(
                        shape=tensor.shape, name=param_name, dtype='float32'))
            return input_spec


def allow_oom(cls):
    def _deco(func):
        def _wrapper(self, *args, **kwargs):
            try:
                func(self, *args, **kwargs)
            except (SystemError, RuntimeError, OSError, MemoryError) as e:
                # XXX: This may not cover all OOM cases.
                msg = str(e)
                if "Out of memory error" in msg \
                    or "(External) CUDNN error(4), CUDNN_STATUS_INTERNAL_ERROR." in msg \
                    or isinstance(e, MemoryError):
                    logging.warning("An OOM error has been ignored.")
                else:
                    raise

        return _wrapper

    for key, value in inspect.getmembers(cls):
        if key.startswith('test'):
            value = _deco(value)
            setattr(cls, key, value)

    return cls


TestModel = _TestModelNamespace.TestModel
