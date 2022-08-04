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

# Refer to
# https://github.com/GeoZcx/A-deeply-supervised-image-fusion-network-for-change-detection-in-remote-sensing-images .

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.vision.models import vgg16

from .layers import Conv1x1, make_norm, ChannelAttention, SpatialAttention


class DSIFN(nn.Layer):
    """
    The DSIFN implementation based on PaddlePaddle.

    The original article refers to
        C. Zhang, et al., "A deeply supervised image fusion network for change detection in high resolution bi-temporal remote 
        sensing images"
        (https://www.sciencedirect.com/science/article/pii/S0924271620301532).

    Note that in this implementation, there is a flexible number of target classes.

    Args:
        num_classes (int): The number of target classes.
        use_dropout (bool, optional): A bool value that indicates whether to use dropout layers. When the model is trained 
            on a relatively small dataset, the dropout layers help prevent overfitting. Default: False.
    """

    def __init__(self, num_classes, use_dropout=False):
        super(DSIFN, self).__init__()

        self.encoder1 = self.encoder2 = VGG16FeaturePicker()

        self.sa1 = SpatialAttention()
        self.sa2 = SpatialAttention()
        self.sa3 = SpatialAttention()
        self.sa4 = SpatialAttention()
        self.sa5 = SpatialAttention()

        self.ca1 = ChannelAttention(in_ch=1024)
        self.bn_ca1 = make_norm(1024)
        self.o1_conv1 = conv2d_bn(1024, 512, use_dropout)
        self.o1_conv2 = conv2d_bn(512, 512, use_dropout)
        self.bn_sa1 = make_norm(512)
        self.o1_conv3 = Conv1x1(512, num_classes)
        self.trans_conv1 = nn.Conv2DTranspose(512, 512, kernel_size=2, stride=2)

        self.ca2 = ChannelAttention(in_ch=1536)
        self.bn_ca2 = make_norm(1536)
        self.o2_conv1 = conv2d_bn(1536, 512, use_dropout)
        self.o2_conv2 = conv2d_bn(512, 256, use_dropout)
        self.o2_conv3 = conv2d_bn(256, 256, use_dropout)
        self.bn_sa2 = make_norm(256)
        self.o2_conv4 = Conv1x1(256, num_classes)
        self.trans_conv2 = nn.Conv2DTranspose(256, 256, kernel_size=2, stride=2)

        self.ca3 = ChannelAttention(in_ch=768)
        self.o3_conv1 = conv2d_bn(768, 256, use_dropout)
        self.o3_conv2 = conv2d_bn(256, 128, use_dropout)
        self.o3_conv3 = conv2d_bn(128, 128, use_dropout)
        self.bn_sa3 = make_norm(128)
        self.o3_conv4 = Conv1x1(128, num_classes)
        self.trans_conv3 = nn.Conv2DTranspose(128, 128, kernel_size=2, stride=2)

        self.ca4 = ChannelAttention(in_ch=384)
        self.o4_conv1 = conv2d_bn(384, 128, use_dropout)
        self.o4_conv2 = conv2d_bn(128, 64, use_dropout)
        self.o4_conv3 = conv2d_bn(64, 64, use_dropout)
        self.bn_sa4 = make_norm(64)
        self.o4_conv4 = Conv1x1(64, num_classes)
        self.trans_conv4 = nn.Conv2DTranspose(64, 64, kernel_size=2, stride=2)

        self.ca5 = ChannelAttention(in_ch=192)
        self.o5_conv1 = conv2d_bn(192, 64, use_dropout)
        self.o5_conv2 = conv2d_bn(64, 32, use_dropout)
        self.o5_conv3 = conv2d_bn(32, 16, use_dropout)
        self.bn_sa5 = make_norm(16)
        self.o5_conv4 = Conv1x1(16, num_classes)

        self.init_weight()

    def forward(self, t1, t2):
        # Extract bi-temporal features.
        with paddle.no_grad():
            self.encoder1.eval(), self.encoder2.eval()
            t1_feats = self.encoder1(t1)
            t2_feats = self.encoder2(t2)

        t1_f_l3, t1_f_l8, t1_f_l15, t1_f_l22, t1_f_l29 = t1_feats
        t2_f_l3, t2_f_l8, t2_f_l15, t2_f_l22, t2_f_l29, = t2_feats

        aux_x = []

        # Multi-level decoding
        x = paddle.concat([t1_f_l29, t2_f_l29], axis=1)
        x = self.o1_conv1(x)
        x = self.o1_conv2(x)
        x = self.sa1(x) * x
        x = self.bn_sa1(x)
        if self.training:
            aux_x.append(x)

        x = self.trans_conv1(x)
        x = paddle.concat([x, t1_f_l22, t2_f_l22], axis=1)
        x = self.ca2(x) * x
        x = self.o2_conv1(x)
        x = self.o2_conv2(x)
        x = self.o2_conv3(x)
        x = self.sa2(x) * x
        x = self.bn_sa2(x)
        if self.training:
            aux_x.append(x)

        x = self.trans_conv2(x)
        x = paddle.concat([x, t1_f_l15, t2_f_l15], axis=1)
        x = self.ca3(x) * x
        x = self.o3_conv1(x)
        x = self.o3_conv2(x)
        x = self.o3_conv3(x)
        x = self.sa3(x) * x
        x = self.bn_sa3(x)
        if self.training:
            aux_x.append(x)

        x = self.trans_conv3(x)
        x = paddle.concat([x, t1_f_l8, t2_f_l8], axis=1)
        x = self.ca4(x) * x
        x = self.o4_conv1(x)
        x = self.o4_conv2(x)
        x = self.o4_conv3(x)
        x = self.sa4(x) * x
        x = self.bn_sa4(x)
        if self.training:
            aux_x.append(x)

        x = self.trans_conv4(x)
        x = paddle.concat([x, t1_f_l3, t2_f_l3], axis=1)
        x = self.ca5(x) * x
        x = self.o5_conv1(x)
        x = self.o5_conv2(x)
        x = self.o5_conv3(x)
        x = self.sa5(x) * x
        x = self.bn_sa5(x)

        out5 = self.o5_conv4(x)

        if not self.training:
            return [out5]
        else:
            size = paddle.shape(t1)[2:]
            out1 = F.interpolate(
                self.o1_conv3(aux_x[0]),
                size=size,
                mode='bilinear',
                align_corners=True)
            out2 = F.interpolate(
                self.o2_conv4(aux_x[1]),
                size=size,
                mode='bilinear',
                align_corners=True)
            out3 = F.interpolate(
                self.o3_conv4(aux_x[2]),
                size=size,
                mode='bilinear',
                align_corners=True)
            out4 = F.interpolate(
                self.o4_conv4(aux_x[3]),
                size=size,
                mode='bilinear',
                align_corners=True)
            return [out5, out4, out3, out2, out1]

    def init_weight(self):
        # Do nothing
        pass


class VGG16FeaturePicker(nn.Layer):
    def __init__(self, indices=(3, 8, 15, 22, 29)):
        super(VGG16FeaturePicker, self).__init__()
        features = list(vgg16(pretrained=True).features)[:30]
        self.features = nn.LayerList(features)
        self.features.eval()
        self.indices = set(indices)

    def forward(self, x):
        picked_feats = []
        for idx, model in enumerate(self.features):
            x = model(x)
            if idx in self.indices:
                picked_feats.append(x)
        return picked_feats


def conv2d_bn(in_ch, out_ch, with_dropout=True):
    lst = [
        nn.Conv2D(
            in_ch, out_ch, kernel_size=3, stride=1, padding=1),
        nn.PReLU(),
        make_norm(out_ch),
    ]
    if with_dropout:
        lst.append(nn.Dropout(p=0.6))
    return nn.Sequential(*lst)
