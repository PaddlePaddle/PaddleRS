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

from paddlers.models.ppdet.modeling import \
                         initializer as init
from paddlers.rs_models.seg.farseg import FPN, \
                         ResNetEncoder,AsymmetricDecoder


def conv_with_kaiming_uniform(use_gn=False, use_relu=False):
    def make_conv(in_channels, out_channels, kernel_size, stride=1, dilation=1):
        conv = nn.Conv2D(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=dilation * (kernel_size - 1) // 2,
            dilation=dilation,
            bias_attr=False if use_gn else True)

        init.kaiming_uniform_(conv.weight, a=1)
        if not use_gn:
            init.constant_(conv.bias, 0)
        module = [conv, ]
        if use_gn:
            raise NotImplementedError
        if use_relu:
            module.append(nn.ReLU())
        if len(module) > 1:
            return nn.Sequential(*module)
        return conv

    return make_conv


default_conv_block = conv_with_kaiming_uniform(use_gn=False, use_relu=False)


class FactSeg(nn.Layer):
    """
     The FactSeg implementation based on PaddlePaddle.

     The original article refers to
     A. Ma, J. Wang, Y. Zhong and Z. Zheng, "FactSeg: Foreground Activation
     -Driven Small Object Semantic Segmentation in Large-Scale Remote Sensing
      Imagery,"in IEEE Transactions on Geoscience and Remote Sensing, vol. 60,
       pp. 1-16, 2022, Art no. 5606216.


     Args:
         in_channels (int): The number of image channels for the input model.
         num_classes (int): The unique number of target classes.
         backbone (str, optional): A backbone network, models available in
         `paddle.vision.models.resnet`. Default: resnet50.
         backbone_pretrained (bool, optional): Whether the backbone network uses
         IMAGENET pretrained weights. Default: True.
     """

    def __init__(self,
                 in_channels,
                 num_classes,
                 backbone='resnet50',
                 backbone_pretrained=True):
        super(FactSeg, self).__init__()
        backbone = backbone.lower()
        self.resencoder = ResNetEncoder(
            backbone=backbone,
            in_channels=in_channels,
            pretrained=backbone_pretrained)
        self.resencoder.resnet._sub_layers.pop('fc')
        self.fgfpn = FPN(in_channels_list=[256, 512, 1024, 2048],
                         out_channels=256,
                         conv_block=default_conv_block)
        self.bifpn = FPN(in_channels_list=[256, 512, 1024, 2048],
                         out_channels=256,
                         conv_block=default_conv_block)
        self.fg_decoder = AsymmetricDecoder(
            in_channels=256,
            out_channels=128,
            in_feature_output_strides=(4, 8, 16, 32),
            out_feature_output_stride=4,
            conv_block=nn.Conv2D)
        self.bi_decoder = AsymmetricDecoder(
            in_channels=256,
            out_channels=128,
            in_feature_output_strides=(4, 8, 16, 32),
            out_feature_output_stride=4,
            conv_block=nn.Conv2D)
        self.fg_cls = nn.Conv2D(128, num_classes, kernel_size=1)
        self.bi_cls = nn.Conv2D(128, 1, kernel_size=1)
        self.config_loss = ['joint_loss']
        self.config_foreground = []
        self.fbattention_atttention = False

    def forward(self, x):
        feat_list = self.resencoder(x)
        if 'skip_decoder' in []:
            fg_out = self.fgskip_deocder(feat_list)
            bi_out = self.bgskip_deocder(feat_list)
        else:
            forefeat_list = list(self.fgfpn(feat_list))
            binaryfeat_list = self.bifpn(feat_list)
            if self.fbattention_atttention:
                for i in range(len(binaryfeat_list)):
                    forefeat_list[i] = self.fbatt_block_list[i](
                        binaryfeat_list[i], forefeat_list[i])
            fg_out = self.fg_decoder(forefeat_list)
            bi_out = self.bi_decoder(binaryfeat_list)
        fg_pred = self.fg_cls(fg_out)
        bi_pred = self.bi_cls(bi_out)
        fg_pred = F.interpolate(
            fg_pred, scale_factor=4.0, mode='bilinear', align_corners=True)
        bi_pred = F.interpolate(
            bi_pred, scale_factor=4.0, mode='bilinear', align_corners=True)
        if self.training:
            return [fg_pred]
        else:
            binary_prob = F.sigmoid(bi_pred)
            cls_prob = F.softmax(fg_pred, axis=1)
            cls_prob[:, 0, :, :] = cls_prob[:, 0, :, :] * (
                1 - binary_prob).squeeze(axis=1)
            cls_prob[:, 1:, :, :] = cls_prob[:, 1:, :, :] * binary_prob
            z = paddle.sum(cls_prob, axis=1)
            z = z.unsqueeze(axis=1)
            cls_prob = paddle.divide(cls_prob, z)
            return [cls_prob]
