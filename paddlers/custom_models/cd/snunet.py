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

# Refer to https://github.com/likyoo/Siam-NestedUNet .

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from .layers import Conv1x1, MaxPool2x2, make_norm, ChannelAttention
from .param_init import KaimingInitMixin


class SNUNet(nn.Layer, KaimingInitMixin):
    """
    The SNUNet implementation based on PaddlePaddle.

    The original article refers to
        S. Fang, et al., "SNUNet-CD: A Densely Connected Siamese Network for Change Detection of VHR Images"
        (https://ieeexplore.ieee.org/document/9355573).

    Note that bilinear interpolation is adopted as the upsampling method, which is different from the paper.

    Args:
        in_channels (int): The number of bands of the input images.
        num_classes (int): The number of target classes.
        width (int, optional): The output channels of the first convolutional layer. Default: 32.
    """

    def __init__(self, in_channels, num_classes, width=32):
        super(SNUNet, self).__init__()

        filters = (width, width * 2, width * 4, width * 8, width * 16)

        self.conv0_0 = ConvBlockNested(in_channels, filters[0], filters[0])
        self.conv1_0 = ConvBlockNested(filters[0], filters[1], filters[1])
        self.conv2_0 = ConvBlockNested(filters[1], filters[2], filters[2])
        self.conv3_0 = ConvBlockNested(filters[2], filters[3], filters[3])
        self.conv4_0 = ConvBlockNested(filters[3], filters[4], filters[4])
        self.down1 = MaxPool2x2()
        self.down2 = MaxPool2x2()
        self.down3 = MaxPool2x2()
        self.down4 = MaxPool2x2()
        self.up1_0 = Up(filters[1])
        self.up2_0 = Up(filters[2])
        self.up3_0 = Up(filters[3])
        self.up4_0 = Up(filters[4])

        self.conv0_1 = ConvBlockNested(filters[0] * 2 + filters[1], filters[0],
                                       filters[0])
        self.conv1_1 = ConvBlockNested(filters[1] * 2 + filters[2], filters[1],
                                       filters[1])
        self.conv2_1 = ConvBlockNested(filters[2] * 2 + filters[3], filters[2],
                                       filters[2])
        self.conv3_1 = ConvBlockNested(filters[3] * 2 + filters[4], filters[3],
                                       filters[3])
        self.up1_1 = Up(filters[1])
        self.up2_1 = Up(filters[2])
        self.up3_1 = Up(filters[3])

        self.conv0_2 = ConvBlockNested(filters[0] * 3 + filters[1], filters[0],
                                       filters[0])
        self.conv1_2 = ConvBlockNested(filters[1] * 3 + filters[2], filters[1],
                                       filters[1])
        self.conv2_2 = ConvBlockNested(filters[2] * 3 + filters[3], filters[2],
                                       filters[2])
        self.up1_2 = Up(filters[1])
        self.up2_2 = Up(filters[2])

        self.conv0_3 = ConvBlockNested(filters[0] * 4 + filters[1], filters[0],
                                       filters[0])
        self.conv1_3 = ConvBlockNested(filters[1] * 4 + filters[2], filters[1],
                                       filters[1])
        self.up1_3 = Up(filters[1])

        self.conv0_4 = ConvBlockNested(filters[0] * 5 + filters[1], filters[0],
                                       filters[0])

        self.ca_intra = ChannelAttention(filters[0], ratio=4)
        self.ca_inter = ChannelAttention(filters[0] * 4, ratio=16)

        self.conv_out = Conv1x1(filters[0] * 4, num_classes)

        self.init_weight()

    def forward(self, t1, t2):
        x0_0_t1 = self.conv0_0(t1)
        x1_0_t1 = self.conv1_0(self.down1(x0_0_t1))
        x2_0_t1 = self.conv2_0(self.down2(x1_0_t1))
        x3_0_t1 = self.conv3_0(self.down3(x2_0_t1))

        x0_0_t2 = self.conv0_0(t2)
        x1_0_t2 = self.conv1_0(self.down1(x0_0_t2))
        x2_0_t2 = self.conv2_0(self.down2(x1_0_t2))
        x3_0_t2 = self.conv3_0(self.down3(x2_0_t2))
        x4_0_t2 = self.conv4_0(self.down4(x3_0_t2))

        x0_1 = self.conv0_1(
            paddle.concat([x0_0_t1, x0_0_t2, self.up1_0(x1_0_t2)], 1))
        x1_1 = self.conv1_1(
            paddle.concat([x1_0_t1, x1_0_t2, self.up2_0(x2_0_t2)], 1))
        x0_2 = self.conv0_2(
            paddle.concat([x0_0_t1, x0_0_t2, x0_1, self.up1_1(x1_1)], 1))

        x2_1 = self.conv2_1(
            paddle.concat([x2_0_t1, x2_0_t2, self.up3_0(x3_0_t2)], 1))
        x1_2 = self.conv1_2(
            paddle.concat([x1_0_t1, x1_0_t2, x1_1, self.up2_1(x2_1)], 1))
        x0_3 = self.conv0_3(
            paddle.concat([x0_0_t1, x0_0_t2, x0_1, x0_2, self.up1_2(x1_2)], 1))

        x3_1 = self.conv3_1(
            paddle.concat([x3_0_t1, x3_0_t2, self.up4_0(x4_0_t2)], 1))
        x2_2 = self.conv2_2(
            paddle.concat([x2_0_t1, x2_0_t2, x2_1, self.up3_1(x3_1)], 1))
        x1_3 = self.conv1_3(
            paddle.concat([x1_0_t1, x1_0_t2, x1_1, x1_2, self.up2_2(x2_2)], 1))
        x0_4 = self.conv0_4(
            paddle.concat(
                [x0_0_t1, x0_0_t2, x0_1, x0_2, x0_3, self.up1_3(x1_3)], 1))

        out = paddle.concat([x0_1, x0_2, x0_3, x0_4], 1)

        intra = paddle.sum(paddle.stack([x0_1, x0_2, x0_3, x0_4]), axis=0)
        m_intra = self.ca_intra(intra)
        out = self.ca_inter(out) * (out + paddle.tile(m_intra, (1, 4, 1, 1)))

        pred = self.conv_out(out)
        return [pred]


class ConvBlockNested(nn.Layer):
    def __init__(self, in_ch, out_ch, mid_ch):
        super(ConvBlockNested, self).__init__()
        self.act = nn.ReLU()
        self.conv1 = nn.Conv2D(in_ch, mid_ch, kernel_size=3, padding=1)
        self.bn1 = make_norm(mid_ch)
        self.conv2 = nn.Conv2D(mid_ch, out_ch, kernel_size=3, padding=1)
        self.bn2 = make_norm(out_ch)

    def forward(self, x):
        x = self.conv1(x)
        identity = x
        x = self.bn1(x)
        x = self.act(x)

        x = self.conv2(x)
        x = self.bn2(x)
        output = self.act(x + identity)
        return output


class Up(nn.Layer):
    def __init__(self, in_ch, use_conv=False):
        super(Up, self).__init__()
        if use_conv:
            self.up = nn.Conv2DTranspose(in_ch, in_ch, 2, stride=2)
        else:
            self.up = nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        x = self.up(x)
        return x
