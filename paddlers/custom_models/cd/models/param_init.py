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

import paddle.nn as nn
import paddle.nn.functional as F


def normal_init(param, *args, **kwargs):
    """
    Initialize parameters with a normal distribution.

    Args:
        param (Tensor): The tensor that needs to be initialized.

    Returns:
        The initialized parameters.
    """

    return nn.initializer.Normal(*args, **kwargs)(param)


def kaiming_normal_init(param, *args, **kwargs):
    """
    Initialize parameters with the Kaiming normal distribution.

    For more information about the Kaiming initialization method, please refer to
        https://arxiv.org/abs/1502.01852

    Args:
        param (Tensor): The tensor that needs to be initialized.

    Returns:
        The initialized parameters.
    """

    return nn.initializer.KaimingNormal(*args, **kwargs)(param)


def constant_init(param, *args, **kwargs):
    """
    Initialize parameters with constants.

    Args:
        param (Tensor): The tensor that needs to be initialized.

    Returns:
        The initialized parameters.
    """

    return nn.initializer.Constant(*args, **kwargs)(param)


class KaimingInitMixin:
    """
    A mix-in that provides the Kaiming initialization functionality.

    Examples:

        from paddlers.custom_models.cd.models.param_init import KaimingInitMixin

        class CustomNet(nn.Layer, KaimingInitMixin):
            def __init__(self, num_channels, num_classes):
                super().__init__()
                self.conv = nn.Conv2D(num_channels, num_classes, 3, 1, 0, bias_attr=False)
                self.bn = nn.BatchNorm2D(num_classes)
                self.init_weight()
    """

    def init_weight(self):
        for layer in self.sublayers():
            if isinstance(layer, nn.Conv2D):
                kaiming_normal_init(layer.weight)
            elif isinstance(layer, (nn.BatchNorm, nn.SyncBatchNorm)):
                constant_init(layer.weight, value=1)
                constant_init(layer.bias, value=0)
