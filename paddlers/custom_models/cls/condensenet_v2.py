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
"""
This code is based on https://github.com/AgentMaker/Paddle-Image-Models
Ths copyright of AgentMaker/Paddle-Image-Models is as follows:
Apache License [see LICENSE for details]
"""

import paddle
import paddle.nn as nn

__all__ = ["CondenseNetV2_a", "CondenseNetV2_b", "CondenseNetV2_c"]


class SELayer(nn.Layer):
    def __init__(self, inplanes, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2D(1)
        self.fc = nn.Sequential(
            nn.Linear(
                inplanes, inplanes // reduction, bias_attr=False),
            nn.ReLU(),
            nn.Linear(
                inplanes // reduction, inplanes, bias_attr=False),
            nn.Sigmoid(), )

    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.avg_pool(x).reshape((b, c))
        y = self.fc(y).reshape((b, c, 1, 1))
        return x * y.expand_as(x)


class HS(nn.Layer):
    def __init__(self):
        super(HS, self).__init__()
        self.relu6 = nn.ReLU6()

    def forward(self, inputs):
        return inputs * self.relu6(inputs + 3) / 6


class Conv(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            groups=1,
            activation="ReLU",
            bn_momentum=0.9, ):
        super(Conv, self).__init__()
        self.add_sublayer(
            "norm", nn.BatchNorm2D(
                in_channels, momentum=bn_momentum))
        if activation == "ReLU":
            self.add_sublayer("activation", nn.ReLU())
        elif activation == "HS":
            self.add_sublayer("activation", HS())
        else:
            raise NotImplementedError
        self.add_sublayer(
            "conv",
            nn.Conv2D(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias_attr=False,
                groups=groups, ), )


def ShuffleLayer(x, groups):
    batchsize, num_channels, height, width = x.shape
    channels_per_group = num_channels // groups
    # reshape
    x = x.reshape((batchsize, groups, channels_per_group, height, width))
    # transpose
    x = x.transpose((0, 2, 1, 3, 4))
    # reshape
    x = x.reshape((batchsize, -1, height, width))
    return x


def ShuffleLayerTrans(x, groups):
    batchsize, num_channels, height, width = x.shape
    channels_per_group = num_channels // groups
    # reshape
    x = x.reshape((batchsize, channels_per_group, groups, height, width))
    # transpose
    x = x.transpose((0, 2, 1, 3, 4))
    # reshape
    x = x.reshape((batchsize, -1, height, width))
    return x


class CondenseLGC(nn.Layer):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            groups=1,
            activation="ReLU", ):
        super(CondenseLGC, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        self.norm = nn.BatchNorm2D(self.in_channels)
        if activation == "ReLU":
            self.activation = nn.ReLU()
        elif activation == "HS":
            self.activation = HS()
        else:
            raise NotImplementedError
        self.conv = nn.Conv2D(
            self.in_channels,
            self.out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=self.groups,
            bias_attr=False, )
        self.register_buffer(
            "index", paddle.zeros(
                (self.in_channels, ), dtype="int64"))

    def forward(self, x):
        x = paddle.index_select(x, self.index, axis=1)
        x = self.norm(x)
        x = self.activation(x)
        x = self.conv(x)
        x = ShuffleLayer(x, self.groups)
        return x


class CondenseSFR(nn.Layer):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            groups=1,
            activation="ReLU", ):
        super(CondenseSFR, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        self.norm = nn.BatchNorm2D(self.in_channels)
        if activation == "ReLU":
            self.activation = nn.ReLU()
        elif activation == "HS":
            self.activation = HS()
        else:
            raise NotImplementedError
        self.conv = nn.Conv2D(
            self.in_channels,
            self.out_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=self.groups,
            bias_attr=False,
            stride=stride, )
        self.register_buffer("index",
                             paddle.zeros(
                                 (self.out_channels, self.out_channels)))

    def forward(self, x):
        x = self.norm(x)
        x = self.activation(x)
        x = ShuffleLayerTrans(x, self.groups)
        x = self.conv(x)  # SIZE: N, C, H, W
        N, C, H, W = x.shape
        x = x.reshape((N, C, H * W))
        x = x.transpose((0, 2, 1))  # SIZE: N, HW, C
        # x SIZE: N, HW, C; self.index SIZE: C, C; OUTPUT SIZE: N, HW, C
        x = paddle.matmul(x, self.index)
        x = x.transpose((0, 2, 1))  # SIZE: N, C, HW
        x = x.reshape((N, C, H, W))  # SIZE: N, C, HW
        return x


class _SFR_DenseLayer(nn.Layer):
    def __init__(
            self,
            in_channels,
            growth_rate,
            group_1x1,
            group_3x3,
            group_trans,
            bottleneck,
            activation,
            use_se=False, ):
        super(_SFR_DenseLayer, self).__init__()
        self.group_1x1 = group_1x1
        self.group_3x3 = group_3x3
        self.group_trans = group_trans
        self.use_se = use_se
        # 1x1 conv i --> b*k
        self.conv_1 = CondenseLGC(
            in_channels,
            bottleneck * growth_rate,
            kernel_size=1,
            groups=self.group_1x1,
            activation=activation, )
        # 3x3 conv b*k --> k
        self.conv_2 = Conv(
            bottleneck * growth_rate,
            growth_rate,
            kernel_size=3,
            padding=1,
            groups=self.group_3x3,
            activation=activation, )
        # 1x1 res conv k(8-16-32)--> i (k*l)
        self.sfr = CondenseSFR(
            growth_rate,
            in_channels,
            kernel_size=1,
            groups=self.group_trans,
            activation=activation, )
        if self.use_se:
            self.se = SELayer(inplanes=growth_rate, reduction=1)

    def forward(self, x):
        x_ = x
        x = self.conv_1(x)
        x = self.conv_2(x)
        if self.use_se:
            x = self.se(x)
        sfr_feature = self.sfr(x)
        y = x_ + sfr_feature
        return paddle.concat([y, x], 1)


class _SFR_DenseBlock(nn.Sequential):
    def __init__(
            self,
            num_layers,
            in_channels,
            growth_rate,
            group_1x1,
            group_3x3,
            group_trans,
            bottleneck,
            activation,
            use_se, ):
        super(_SFR_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _SFR_DenseLayer(
                in_channels + i * growth_rate,
                growth_rate,
                group_1x1,
                group_3x3,
                group_trans,
                bottleneck,
                activation,
                use_se, )
            self.add_sublayer("denselayer_%d" % (i + 1), layer)


class _Transition(nn.Layer):
    def __init__(self):
        super(_Transition, self).__init__()
        self.pool = nn.AvgPool2D(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.pool(x)
        return x


class CondenseNetV2(nn.Layer):
    def __init__(
            self,
            stages,
            growth,
            HS_start_block,
            SE_start_block,
            fc_channel,
            group_1x1,
            group_3x3,
            group_trans,
            bottleneck,
            last_se_reduction,
            in_channels=3,
            class_num=1000, ):
        super(CondenseNetV2, self).__init__()
        self.stages = stages
        self.growth = growth
        self.in_channels = in_channels
        self.class_num = class_num
        self.last_se_reduction = last_se_reduction
        assert len(self.stages) == len(self.growth)
        self.progress = 0.0

        self.init_stride = 2
        self.pool_size = 7

        self.features = nn.Sequential()
        # Initial nChannels should be 3
        self.num_features = 2 * self.growth[0]
        # Dense-block 1 (224x224)
        self.features.add_sublayer(
            "init_conv",
            nn.Conv2D(
                in_channels,
                self.num_features,
                kernel_size=3,
                stride=self.init_stride,
                padding=1,
                bias_attr=False, ), )
        for i in range(len(self.stages)):
            activation = "HS" if i >= HS_start_block else "ReLU"
            use_se = True if i >= SE_start_block else False
            # Dense-block i
            self.add_block(i, group_1x1, group_3x3, group_trans, bottleneck,
                           activation, use_se)

        self.fc = nn.Linear(self.num_features, fc_channel)
        self.fc_act = HS()

        # Classifier layer
        if class_num > 0:
            self.classifier = nn.Linear(fc_channel, class_num)
        self._initialize()

    def add_block(self, i, group_1x1, group_3x3, group_trans, bottleneck,
                  activation, use_se):
        # Check if ith is the last one
        last = i == len(self.stages) - 1
        block = _SFR_DenseBlock(
            num_layers=self.stages[i],
            in_channels=self.num_features,
            growth_rate=self.growth[i],
            group_1x1=group_1x1,
            group_3x3=group_3x3,
            group_trans=group_trans,
            bottleneck=bottleneck,
            activation=activation,
            use_se=use_se, )
        self.features.add_sublayer("denseblock_%d" % (i + 1), block)
        self.num_features += self.stages[i] * self.growth[i]
        if not last:
            trans = _Transition()
            self.features.add_sublayer("transition_%d" % (i + 1), trans)
        else:
            self.features.add_sublayer("norm_last",
                                       nn.BatchNorm2D(self.num_features))
            self.features.add_sublayer("relu_last", nn.ReLU())
            self.features.add_sublayer("pool_last",
                                       nn.AvgPool2D(self.pool_size))
            # if useSE:
            self.features.add_sublayer(
                "se_last",
                SELayer(
                    self.num_features, reduction=self.last_se_reduction))

    def forward(self, x):
        features = self.features(x)
        out = features.reshape((features.shape[0], -1))
        out = self.fc(out)
        out = self.fc_act(out)

        if self.class_num > 0:
            out = self.classifier(out)

        return out

    def _initialize(self):
        # initialize
        for m in self.sublayers():
            if isinstance(m, nn.Conv2D):
                nn.initializer.KaimingNormal()(m.weight)
            elif isinstance(m, nn.BatchNorm2D):
                nn.initializer.Constant(value=1.0)(m.weight)
                nn.initializer.Constant(value=0.0)(m.bias)


def CondenseNetV2_a(**kwargs):
    model = CondenseNetV2(
        stages=[1, 1, 4, 6, 8],
        growth=[8, 8, 16, 32, 64],
        HS_start_block=2,
        SE_start_block=3,
        fc_channel=828,
        group_1x1=8,
        group_3x3=8,
        group_trans=8,
        bottleneck=4,
        last_se_reduction=16,
        **kwargs)
    return model


def CondenseNetV2_b(**kwargs):
    model = CondenseNetV2(
        stages=[2, 4, 6, 8, 6],
        growth=[6, 12, 24, 48, 96],
        HS_start_block=2,
        SE_start_block=3,
        fc_channel=1024,
        group_1x1=6,
        group_3x3=6,
        group_trans=6,
        bottleneck=4,
        last_se_reduction=16,
        **kwargs)
    return model


def CondenseNetV2_c(**kwargs):
    model = CondenseNetV2(
        stages=[4, 6, 8, 10, 8],
        growth=[8, 16, 32, 64, 128],
        HS_start_block=2,
        SE_start_block=3,
        fc_channel=1024,
        group_1x1=8,
        group_3x3=8,
        group_trans=8,
        bottleneck=4,
        last_se_reduction=16,
        **kwargs)
    return model
