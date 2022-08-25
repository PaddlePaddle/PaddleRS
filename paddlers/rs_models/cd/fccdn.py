# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from .layers import BasicConv, MaxPool2x2, Conv1x1, Conv3x3

bn_mom = 1 - 0.0003


class NLBlock(nn.Layer):
    def __init__(self, in_channels):
        super(NLBlock, self).__init__()
        self.conv_v = BasicConv(
            in_ch=in_channels,
            out_ch=in_channels,
            kernel_size=3,
            norm=nn.BatchNorm2D(
                in_channels, momentum=0.9))
        self.W = BasicConv(
            in_ch=in_channels,
            out_ch=in_channels,
            kernel_size=3,
            norm=nn.BatchNorm2D(
                in_channels, momentum=0.9),
            act=nn.ReLU())

    def forward(self, x):
        batch_size, c, h, w = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
        value = self.conv_v(x)
        value = value.reshape([batch_size, c, value.shape[2] * value.shape[3]])
        value = value.transpose([0, 2, 1])  # B * (H*W) * value_channels
        key = x.reshape([batch_size, c, h * w])  # B * key_channels * (H*W)
        query = x.reshape([batch_size, c, h * w])
        query = query.transpose([0, 2, 1])

        sim_map = paddle.matmul(query, key)  # B * (H*W) * (H*W)
        sim_map = (c**-.5) * sim_map  # B * (H*W) * (H*W)
        sim_map = nn.functional.softmax(sim_map, axis=-1)  # B * (H*W) * (H*W)

        context = paddle.matmul(sim_map, value)
        context = context.transpose([0, 2, 1])
        context = context.reshape([batch_size, c, *x.shape[2:]])
        context = self.W(context)

        return context


class NLFPN(nn.Layer):
    """ Non-local feature parymid network"""

    def __init__(self, in_dim, reduction=True):
        super(NLFPN, self).__init__()
        if reduction:
            self.reduction = BasicConv(
                in_ch=in_dim,
                out_ch=in_dim // 4,
                kernel_size=1,
                norm=nn.BatchNorm2D(
                    in_dim // 4, momentum=bn_mom),
                act=nn.ReLU())
            self.re_reduction = BasicConv(
                in_ch=in_dim // 4,
                out_ch=in_dim,
                kernel_size=1,
                norm=nn.BatchNorm2D(
                    in_dim, momentum=bn_mom),
                act=nn.ReLU())
            in_dim = in_dim // 4
        else:
            self.reduction = None
            self.re_reduction = None
        self.conv_e1 = BasicConv(
            in_dim,
            in_dim,
            kernel_size=3,
            norm=nn.BatchNorm2D(
                in_dim, momentum=bn_mom),
            act=nn.ReLU())
        self.conv_e2 = BasicConv(
            in_dim,
            in_dim * 2,
            kernel_size=3,
            norm=nn.BatchNorm2D(
                in_dim * 2, momentum=bn_mom),
            act=nn.ReLU())
        self.conv_e3 = BasicConv(
            in_dim * 2,
            in_dim * 4,
            kernel_size=3,
            norm=nn.BatchNorm2D(
                in_dim * 4, momentum=bn_mom),
            act=nn.ReLU())
        self.conv_d1 = BasicConv(
            in_dim,
            in_dim,
            kernel_size=3,
            norm=nn.BatchNorm2D(
                in_dim, momentum=bn_mom),
            act=nn.ReLU())
        self.conv_d2 = BasicConv(
            in_dim * 2,
            in_dim,
            kernel_size=3,
            norm=nn.BatchNorm2D(
                in_dim, momentum=bn_mom),
            act=nn.ReLU())
        self.conv_d3 = BasicConv(
            in_dim * 4,
            in_dim * 2,
            kernel_size=3,
            norm=nn.BatchNorm2D(
                in_dim * 2, momentum=bn_mom),
            act=nn.ReLU())
        self.nl3 = NLBlock(in_dim * 2)
        self.nl2 = NLBlock(in_dim)
        self.nl1 = NLBlock(in_dim)

        self.downsample_x2 = nn.MaxPool2D(stride=2, kernel_size=2)
        self.upsample_x2 = nn.UpsamplingBilinear2D(scale_factor=2)

    def forward(self, x):
        if self.reduction is not None:
            x = self.reduction(x)
        e1 = self.conv_e1(x)  # C,H,W
        e2 = self.conv_e2(self.downsample_x2(e1))  # 2C,H/2,W/2
        e3 = self.conv_e3(self.downsample_x2(e2))  # 4C,H/4,W/4

        d3 = self.conv_d3(e3)  # 2C,H/4,W/4
        nl = self.nl3(d3)
        d3 = self.upsample_x2(paddle.multiply(d3, nl))  ##2C,H/2,W/2
        d2 = self.conv_d2(e2 + d3)  # C,H/2,W/2
        nl = self.nl2(d2)
        d2 = self.upsample_x2(paddle.multiply(d2, nl))  # C,H,W
        d1 = self.conv_d1(e1 + d2)
        nl = self.nl1(d1)
        d1 = paddle.multiply(d1, nl)  # C,H,W
        if self.re_reduction is not None:
            d1 = self.re_reduction(d1)

        return d1


class Cat(nn.Layer):
    def __init__(self, in_chn_high, in_chn_low, out_chn, upsample=False):
        super(Cat, self).__init__()
        self.do_upsample = upsample
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv2d = BasicConv(
            in_chn_high + in_chn_low,
            out_chn,
            kernel_size=1,
            norm=nn.BatchNorm2D(
                out_chn, momentum=bn_mom),
            act=nn.ReLU())

    def forward(self, x, y):
        if self.do_upsample:
            x = self.upsample(x)

        x = paddle.concat((x, y), 1)

        return self.conv2d(x)


class DoubleConv(nn.Layer):
    def __init__(self, in_chn, out_chn, stride=1, dilation=1):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2D(
                in_chn,
                out_chn,
                kernel_size=3,
                stride=stride,
                dilation=dilation,
                padding=dilation),
            nn.BatchNorm2D(
                out_chn, momentum=bn_mom),
            nn.ReLU(),
            nn.Conv2D(
                out_chn, out_chn, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2D(
                out_chn, momentum=bn_mom),
            nn.ReLU())

    def forward(self, x):
        x = self.conv(x)
        return x


class SEModule(nn.Layer):
    def __init__(self, channels, reduction_channels):
        super(SEModule, self).__init__()
        self.fc1 = nn.Conv2D(
            channels,
            reduction_channels,
            kernel_size=1,
            padding=0,
            bias_attr=True)
        self.ReLU = nn.ReLU()
        self.fc2 = nn.Conv2D(
            reduction_channels,
            channels,
            kernel_size=1,
            padding=0,
            bias_attr=True)

    def forward(self, x):
        x_se = x.reshape(
            [x.shape[0], x.shape[1], x.shape[2] * x.shape[3]]).mean(-1).reshape(
                [x.shape[0], x.shape[1], 1, 1])

        x_se = self.fc1(x_se)
        x_se = self.ReLU(x_se)
        x_se = self.fc2(x_se)
        return x * F.sigmoid(x_se)


class BasicBlock(nn.Layer):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 downsample=None,
                 use_se=False,
                 stride=1,
                 dilation=1):
        super(BasicBlock, self).__init__()
        first_planes = planes
        outplanes = planes * self.expansion

        self.conv1 = DoubleConv(inplanes, first_planes)
        self.conv2 = DoubleConv(
            first_planes, outplanes, stride=stride, dilation=dilation)
        self.se = SEModule(outplanes, planes // 4) if use_se else None
        self.downsample = MaxPool2x2() if downsample else None
        self.ReLU = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        residual = out
        out = self.conv2(out)

        if self.se is not None:
            out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(residual)

        out = out + residual
        out = self.ReLU(out)
        return out


class DenseCatAdd(nn.Layer):
    def __init__(self, in_chn, out_chn):
        super(DenseCatAdd, self).__init__()
        self.conv1 = BasicConv(in_chn, in_chn, kernel_size=3, act=nn.ReLU())
        self.conv2 = BasicConv(in_chn, in_chn, kernel_size=3, act=nn.ReLU())
        self.conv3 = BasicConv(in_chn, in_chn, kernel_size=3, act=nn.ReLU())
        self.conv_out = BasicConv(
            in_chn,
            out_chn,
            kernel_size=1,
            norm=nn.BatchNorm2D(
                out_chn, momentum=bn_mom),
            act=nn.ReLU())

    def forward(self, x, y):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2 + x1)

        y1 = self.conv1(y)
        y2 = self.conv2(y1)
        y3 = self.conv3(y2 + y1)

        return self.conv_out(x1 + x2 + x3 + y1 + y2 + y3)


class DenseCatDiff(nn.Layer):
    def __init__(self, in_chn, out_chn):
        super(DenseCatDiff, self).__init__()
        self.conv1 = BasicConv(in_chn, in_chn, kernel_size=3, act=nn.ReLU())
        self.conv2 = BasicConv(in_chn, in_chn, kernel_size=3, act=nn.ReLU())
        self.conv3 = BasicConv(in_chn, in_chn, kernel_size=3, act=nn.ReLU())
        self.conv_out = BasicConv(
            in_ch=in_chn,
            out_ch=out_chn,
            kernel_size=1,
            norm=nn.BatchNorm2D(
                out_chn, momentum=bn_mom),
            act=nn.ReLU())

    def forward(self, x, y):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2 + x1)

        y1 = self.conv1(y)
        y2 = self.conv2(y1)
        y3 = self.conv3(y2 + y1)
        out = self.conv_out(paddle.abs(x1 + x2 + x3 - y1 - y2 - y3))
        return out


class DFModule(nn.Layer):
    """Dense connection-based feature fusion module"""

    def __init__(self, dim_in, dim_out, reduction=True):
        super(DFModule, self).__init__()
        if reduction:
            self.reduction = Conv1x1(
                dim_in,
                dim_in // 2,
                norm=nn.BatchNorm2D(
                    dim_in // 2, momentum=bn_mom),
                act=nn.ReLU())
            dim_in = dim_in // 2
        else:
            self.reduction = None
        self.cat1 = DenseCatAdd(dim_in, dim_out)
        self.cat2 = DenseCatDiff(dim_in, dim_out)
        self.conv1 = Conv3x3(
            dim_out,
            dim_out,
            norm=nn.BatchNorm2D(
                dim_out, momentum=bn_mom),
            act=nn.ReLU())

    def forward(self, x1, x2):
        if self.reduction is not None:
            x1 = self.reduction(x1)
            x2 = self.reduction(x2)
        x_add = self.cat1(x1, x2)
        x_diff = self.cat2(x1, x2)
        y = self.conv1(x_diff) + x_add
        return y


class FCCDN(nn.Layer):
    """
    The FCCDN implementation based on PaddlePaddle.

    The original article refers to
        Pan Chen, et al., "FCCDN: Feature Constraint Network for VHR Image Change Detection"
        (https://arxiv.org/pdf/2105.10860.pdf).

    Args:
        in_channels (int): Number of input channels. Default: 3.
        num_classes (int): Number of target classes. Default: 2.
        os (int): Number of output stride. Default: 16.
        use_se (bool): Whether to use SEModule. Default: True.
    """

    def __init__(self, in_channels=3, num_classes=2, os=16, use_se=True):
        super(FCCDN, self).__init__()
        if os >= 16:
            dilation_list = [1, 1, 1, 1]
            stride_list = [2, 2, 2, 2]
            pool_list = [True, True, True, True]
        elif os == 8:
            dilation_list = [2, 1, 1, 1]
            stride_list = [1, 2, 2, 2]
            pool_list = [False, True, True, True]
        else:
            dilation_list = [2, 2, 1, 1]
            stride_list = [1, 1, 2, 2]
            pool_list = [False, False, True, True]
        se_list = [use_se, use_se, use_se, use_se]
        channel_list = [256, 128, 64, 32]
        # Encoder
        self.block1 = BasicBlock(in_channels, channel_list[3], pool_list[3],
                                 se_list[3], stride_list[3], dilation_list[3])
        self.block2 = BasicBlock(channel_list[3], channel_list[2], pool_list[2],
                                 se_list[2], stride_list[2], dilation_list[2])
        self.block3 = BasicBlock(channel_list[2], channel_list[1], pool_list[1],
                                 se_list[1], stride_list[1], dilation_list[1])
        self.block4 = BasicBlock(channel_list[1], channel_list[0], pool_list[0],
                                 se_list[0], stride_list[0], dilation_list[0])

        # Center
        self.center = NLFPN(channel_list[0], True)

        # Decoder
        self.decoder3 = Cat(channel_list[0],
                            channel_list[1],
                            channel_list[1],
                            upsample=pool_list[0])
        self.decoder2 = Cat(channel_list[1],
                            channel_list[2],
                            channel_list[2],
                            upsample=pool_list[1])
        self.decoder1 = Cat(channel_list[2],
                            channel_list[3],
                            channel_list[3],
                            upsample=pool_list[2])

        self.df1 = DFModule(channel_list[3], channel_list[3], True)
        self.df2 = DFModule(channel_list[2], channel_list[2], True)
        self.df3 = DFModule(channel_list[1], channel_list[1], True)
        self.df4 = DFModule(channel_list[0], channel_list[0], True)

        self.catc3 = Cat(channel_list[0],
                         channel_list[1],
                         channel_list[1],
                         upsample=pool_list[0])
        self.catc2 = Cat(channel_list[1],
                         channel_list[2],
                         channel_list[2],
                         upsample=pool_list[1])
        self.catc1 = Cat(channel_list[2],
                         channel_list[3],
                         channel_list[3],
                         upsample=pool_list[2])

        self.upsample_x2 = nn.Sequential(
            nn.Conv2D(
                channel_list[3], 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2D(
                8, momentum=bn_mom),
            nn.ReLU(),
            nn.UpsamplingBilinear2D(scale_factor=2))

        self.conv_out = nn.Conv2D(
            8, num_classes, kernel_size=3, stride=1, padding=1)
        self.conv_out_class = nn.Conv2D(
            channel_list[3], 1, kernel_size=1, stride=1, padding=0)

    def forward(self, t1, t2):
        e1_1 = self.block1(t1)
        e2_1 = self.block2(e1_1)
        e3_1 = self.block3(e2_1)
        y1 = self.block4(e3_1)

        e1_2 = self.block1(t2)
        e2_2 = self.block2(e1_2)
        e3_2 = self.block3(e2_2)
        y2 = self.block4(e3_2)

        y1 = self.center(y1)
        y2 = self.center(y2)
        c = self.df4(y1, y2)

        y1 = self.decoder3(y1, e3_1)
        y2 = self.decoder3(y2, e3_2)
        c = self.catc3(c, self.df3(y1, y2))

        y1 = self.decoder2(y1, e2_1)
        y2 = self.decoder2(y2, e2_2)
        c = self.catc2(c, self.df2(y1, y2))

        y1 = self.decoder1(y1, e1_1)
        y2 = self.decoder1(y2, e1_2)

        c = self.catc1(c, self.df1(y1, y2))
        y = self.conv_out(self.upsample_x2(c))

        if self.training:
            y1 = self.conv_out_class(y1)
            y2 = self.conv_out_class(y2)
            return [y, [y1, y2]]
        else:
            return [y]
