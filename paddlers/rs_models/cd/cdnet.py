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

from .layers import Conv7x7


class CDNet(nn.Layer):
    """
    The CDNet implementation based on PaddlePaddle.

    The original article refers to
        Pablo F. Alcantarilla, et al., "Street-View Change Detection with Deconvolut
        ional Networks"
        (https://link.springer.com/article/10.1007/s10514-018-9734-5).

    Args:
        in_channels (int): Number of bands of the input images.
        num_classes (int): Number of target classes.
    """

    def __init__(self, in_channels, num_classes):
        super(CDNet, self).__init__()
        self.conv1 = Conv7x7(in_channels, 64, norm=True, act=True)
        self.pool1 = nn.MaxPool2D(2, 2, return_mask=True)
        self.conv2 = Conv7x7(64, 64, norm=True, act=True)
        self.pool2 = nn.MaxPool2D(2, 2, return_mask=True)
        self.conv3 = Conv7x7(64, 64, norm=True, act=True)
        self.pool3 = nn.MaxPool2D(2, 2, return_mask=True)
        self.conv4 = Conv7x7(64, 64, norm=True, act=True)
        self.pool4 = nn.MaxPool2D(2, 2, return_mask=True)
        self.conv5 = Conv7x7(64, 64, norm=True, act=True)
        self.upool4 = nn.MaxUnPool2D(2, 2)
        self.conv6 = Conv7x7(64, 64, norm=True, act=True)
        self.upool3 = nn.MaxUnPool2D(2, 2)
        self.conv7 = Conv7x7(64, 64, norm=True, act=True)
        self.upool2 = nn.MaxUnPool2D(2, 2)
        self.conv8 = Conv7x7(64, 64, norm=True, act=True)
        self.upool1 = nn.MaxUnPool2D(2, 2)
        self.conv_out = Conv7x7(64, num_classes, norm=False, act=False)

    def forward(self, t1, t2):
        x = paddle.concat([t1, t2], axis=1)
        x, ind1 = self.pool1(self.conv1(x))
        x, ind2 = self.pool2(self.conv2(x))
        x, ind3 = self.pool3(self.conv3(x))
        x, ind4 = self.pool4(self.conv4(x))
        x = self.conv5(self.upool4(x, ind4))
        x = self.conv6(self.upool3(x, ind3))
        x = self.conv7(self.upool2(x, ind2))
        x = self.conv8(self.upool1(x, ind1))
        return [self.conv_out(x)]
