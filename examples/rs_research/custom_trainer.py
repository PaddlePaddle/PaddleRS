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
import paddlers
from paddlers.tasks.change_detector import BaseChangeDetector

from attach_tools import Attach

attach = Attach.to(paddlers.tasks.change_detector)


def make_trainer(net_type, attach_trainer=True):
    def _init_func(self,
                   num_classes=2,
                   use_mixed_loss=False,
                   losses=None,
                   **_params_):
        sig = inspect.signature(net_type.__init__)
        net_params = {
            k: p.default
            for k, p in sig.parameters.items() if not p.default is p.empty
        }
        net_params.pop('self', None)
        net_params.pop('num_classes', None)
        # Special rule to parse arguments from `_params_`.
        # When using pdrs.tasks.load_model, `_params_`` is a dict with the key '_params_'.
        # This bypasses the dynamic modification/creation of function signature.
        if '_params_' not in _params_:
            net_params.update(_params_)
        else:
            net_params.update(_params_['_params_'])

        super(trainer_type, self).__init__(
            model_name=net_type.__name__,
            num_classes=num_classes,
            use_mixed_loss=use_mixed_loss,
            losses=losses,
            **net_params)

    if not issubclass(net_type, paddle.nn.Layer):
        raise TypeError("net must be a subclass of paddle.nn.Layer")

    trainer_name = net_type.__name__

    trainer_type = type(trainer_name, (BaseChangeDetector, ),
                        {'__init__': _init_func})
    if attach_trainer:
        trainer_type = attach(trainer_type)
    return trainer_type


def make_trainer_and_build(net_type, *args, **kwargs):
    trainer_type = make_trainer(net_type, attach_trainer=True)
    return trainer_type(*args, **kwargs)


@attach
class CustomTrainer(BaseChangeDetector):
    def __init__(self,
                 num_classes=2,
                 use_mixed_loss=False,
                 losses=None,
                 in_channels=3,
                 att_types='ct',
                 use_dropout=False,
                 **params):
        params.update({
            'in_channels': in_channels,
            'att_types': att_types,
            'use_dropout': use_dropout
        })
        super().__init__(
            model_name='CustomModel',
            num_classes=num_classes,
            use_mixed_loss=use_mixed_loss,
            losses=losses,
            **params)
