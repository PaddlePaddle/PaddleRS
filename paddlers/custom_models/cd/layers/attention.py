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

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from .blocks import Conv1x1, BasicConv


class ChannelAttention(nn.Layer):
    """
    The channel attention module implementation based on PaddlePaddle.

    The original article refers to 
        Sanghyun Woo, et al., "CBAM: Convolutional Block Attention Module"
        (https://arxiv.org/abs/1807.06521).

    Args:
        in_ch (int): The number of channels of the input features.
        ratio (int, optional): The channel reduction ratio. Default: 8.
    """

    def __init__(self, in_ch, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2D(1)
        self.max_pool = nn.AdaptiveMaxPool2D(1)
        self.fc1 = Conv1x1(in_ch, in_ch // ratio, bias=False, act=True)
        self.fc2 = Conv1x1(in_ch // ratio, in_ch, bias=False)

    def forward(self, x):
        avg_out = self.fc2(self.fc1(self.avg_pool(x)))
        max_out = self.fc2(self.fc1(self.max_pool(x)))
        out = avg_out + max_out
        return F.sigmoid(out)


class SpatialAttention(nn.Layer):
    """
    The spatial attention module implementation based on PaddlePaddle.

    The original article refers to 
        Sanghyun Woo, et al., "CBAM: Convolutional Block Attention Module"
        (https://arxiv.org/abs/1807.06521).

    Args:
        kernel_size (int, optional): The size of the convolutional kernel. Default: 7.
    """

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = BasicConv(2, 1, kernel_size, bias=False)

    def forward(self, x):
        avg_out = paddle.mean(x, axis=1, keepdim=True)
        max_out = paddle.max(x, axis=1, keepdim=True)
        x = paddle.concat([avg_out, max_out], axis=1)
        x = self.conv(x)
        return F.sigmoid(x)


class CBAM(nn.Layer):
    """
    The CBAM implementation based on PaddlePaddle.

    The original article refers to 
        Sanghyun Woo, et al., "CBAM: Convolutional Block Attention Module"
        (https://arxiv.org/abs/1807.06521).

    Args:
        in_ch (int): The number of channels of the input features.
        ratio (int, optional): The channel reduction ratio for the channel attention module. Default: 8.
        kernel_size (int, optional): The size of the convolutional kernel used in the spatial attention module. Default: 7.
    """

    def __init__(self, in_ch, ratio=8, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_ch, ratio=ratio)
        self.sa = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        y = self.ca(x) * x
        y = self.sa(y) * y
        return y
