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

# This code is based on https://github.com/Z-Zheng/FarSeg
# The copyright of Z-Zheng/FarSeg is as follows:
# Apache License (see https://github.com/Z-Zheng/FarSeg/blob/master/LICENSE for details).

import math

import paddle.nn as nn
import paddle.nn.functional as F
from paddle.vision.models import resnet

from paddlers.models.ppdet.modeling import initializer as init


class FPNConvBlock(nn.Conv2D):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dilation=1):
        super(FPNConvBlock, self).__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=dilation * (kernel_size - 1) // 2,
            dilation=dilation)
        init.kaiming_uniform_(self.weight, a=1)
        init.constant_(self.bias, value=0)


class DefaultConvBlock(nn.Conv2D):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 bias_attr=None):
        super(DefaultConvBlock, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias_attr=bias_attr)
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)


class ResNetEncoder(nn.Layer):
    def __init__(self, backbone='resnet50', in_channels=3, pretrained=True):
        super(ResNetEncoder, self).__init__()
        self.resnet = getattr(resnet, backbone)(pretrained=pretrained)
        if in_channels != 3:
            self.resnet.conv1 = nn.Conv2D(
                in_channels, 64, 7, stride=2, padding=3, bias_attr=False)

        for layer in self.resnet.sublayers():
            if isinstance(layer, (nn.BatchNorm2D, nn.SyncBatchNorm)):
                layer._momentum = 0.1

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        c2 = self.resnet.layer1(x)
        c3 = self.resnet.layer2(c2)
        c4 = self.resnet.layer3(c3)
        c5 = self.resnet.layer4(c4)

        return [c2, c3, c4, c5]


class FPN(nn.Layer):
    def __init__(self, in_channels_list, out_channels, conv_block=FPNConvBlock):
        super(FPN, self).__init__()

        inner_blocks = []
        layer_blocks = []
        for idx, in_channels in enumerate(in_channels_list, 1):
            if in_channels == 0:
                continue
            inner_blocks.append(conv_block(in_channels, out_channels, 1))
            layer_blocks.append(conv_block(out_channels, out_channels, 3, 1))
        self.inner_blocks = nn.LayerList(inner_blocks)
        self.layer_blocks = nn.LayerList(layer_blocks)

    def forward(self, x):
        last_inner = self.inner_blocks[-1](x[-1])
        results = [self.layer_blocks[-1](last_inner)]
        for i, feature in enumerate(x[-2::-1]):
            inner_block = self.inner_blocks[len(self.inner_blocks) - 2 - i]
            layer_block = self.layer_blocks[len(self.layer_blocks) - 2 - i]
            inner_top_down = F.interpolate(
                last_inner, scale_factor=2, mode="nearest")
            inner_lateral = inner_block(feature)
            last_inner = inner_lateral + inner_top_down
            results.insert(0, layer_block(last_inner))
        return tuple(results)


class FSRelation(nn.Layer):
    def __init__(self,
                 in_channels,
                 channels_list,
                 out_channels,
                 scale_aware_proj=True,
                 conv_block=DefaultConvBlock):
        super(FSRelation, self).__init__()

        self.scale_aware_proj = scale_aware_proj
        if self.scale_aware_proj:
            self.scene_encoder = nn.LayerList([
                nn.Sequential(
                    conv_block(in_channels, out_channels, 1),
                    nn.ReLU(), conv_block(out_channels, out_channels, 1))
                for _ in range(len(channels_list))
            ])
        else:
            self.scene_encoder = nn.Sequential(
                conv_block(in_channels, out_channels, 1),
                nn.ReLU(), conv_block(out_channels, out_channels, 1))

        self.content_encoders = nn.LayerList()
        self.feature_reencoders = nn.LayerList()
        for channel in channels_list:
            self.content_encoders.append(
                nn.Sequential(
                    conv_block(
                        channel, out_channels, 1, bias_attr=True),
                    nn.BatchNorm2D(
                        out_channels, momentum=0.1),
                    nn.ReLU()))
            self.feature_reencoders.append(
                nn.Sequential(
                    conv_block(
                        channel, out_channels, 1, bias_attr=True),
                    nn.BatchNorm2D(
                        out_channels, momentum=0.1),
                    nn.ReLU()))

        self.normalizer = nn.Sigmoid()

    def forward(self, scene_feature, feature_list):
        content_feats = [
            c_en(p_feat)
            for c_en, p_feat in zip(self.content_encoders, feature_list)
        ]
        if self.scale_aware_proj:
            scene_feats = [op(scene_feature) for op in self.scene_encoder]
            relations = [
                self.normalizer((sf * cf).sum(axis=1, keepdim=True))
                for sf, cf in zip(scene_feats, content_feats)
            ]
        else:
            scene_feat = self.scene_encoder(scene_feature)
            relations = [
                self.normalizer((scene_feat * cf).sum(axis=1, keepdim=True))
                for cf in content_feats
            ]
        p_feats = [
            op(p_feat)
            for op, p_feat in zip(self.feature_reencoders, feature_list)
        ]
        refined_feats = [r * p for r, p in zip(relations, p_feats)]
        return refined_feats


class AsymmetricDecoder(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 in_feature_output_strides=(4, 8, 16, 32),
                 out_feature_output_stride=4,
                 conv_block=DefaultConvBlock):
        super(AsymmetricDecoder, self).__init__()

        self.blocks = nn.LayerList()
        for in_feature_output_stride in in_feature_output_strides:
            num_upsample = int(math.log2(int(in_feature_output_stride))) - int(
                math.log2(int(out_feature_output_stride)))
            num_layers = num_upsample if num_upsample != 0 else 1
            self.blocks.append(
                nn.Sequential(* [
                    nn.Sequential(
                        conv_block(
                            in_channels if idx == 0 else out_channels,
                            out_channels,
                            3,
                            1,
                            1,
                            bias_attr=False),
                        nn.BatchNorm2D(
                            out_channels, momentum=0.1),
                        nn.ReLU(),
                        nn.UpsamplingBilinear2D(scale_factor=2) if num_upsample
                        != 0 else nn.Identity(), ) for idx in range(num_layers)
                ]))

    def forward(self, feature_list):
        inner_feature_list = []
        for idx, block in enumerate(self.blocks):
            decoder_feature = block(feature_list[idx])
            inner_feature_list.append(decoder_feature)
        out_feature = sum(inner_feature_list) / len(inner_feature_list)
        return out_feature


class FarSeg(nn.Layer):
    """
    The FarSeg implementation based on PaddlePaddle.

    The original article refers to
    Zheng Z, Zhong Y, Wang J, et al. Foreground-aware relation network for geospatial object segmentation in
    high spatial resolution remote sensing imagery[C]//Proceedings of the IEEE/CVF conference on computer vision
    and pattern recognition. 2020: 4096-4105.

    Args:
        in_channels (int): The number of image channels for the input model. Default: 3.
        num_classes (int): The unique number of target classes. Default: 16.
        backbone (str): A backbone network, models available in `paddle.vision.models.resnet`. Default: resnet50.
        backbone_pretrained (bool): Whether the backbone network uses IMAGENET pretrained weights. Default: True.
        fpn_out_channels (int): The number of channels output by the feature pyramid network. Default: 256.
        fsr_out_channels (int): The number of channels output by the F-S relation module. Default: 256.
        scale_aware_proj (bool): Whether to use scale awareness in F-S relation module. Default: True.
        decoder_out_channels (int): The number of channels output by the decoder. Default: 128.
    """

    def __init__(self,
                 in_channels=3,
                 num_classes=16,
                 backbone='resnet50',
                 backbone_pretrained=True,
                 fpn_out_channels=256,
                 fsr_out_channels=256,
                 scale_aware_proj=True,
                 decoder_out_channels=128):
        super(FarSeg, self).__init__()

        backbone = backbone.lower()
        self.encoder = ResNetEncoder(
            backbone=backbone,
            in_channels=in_channels,
            pretrained=backbone_pretrained)

        fpn_max_in_channels = 2048
        if backbone in ['resnet18', 'resnet34']:
            fpn_max_in_channels = 512
        self.fpn = FPN(in_channels_list=[
            fpn_max_in_channels // (2**(3 - i)) for i in range(4)
        ],
                       out_channels=fpn_out_channels)
        self.gap = nn.AdaptiveAvgPool2D(1)
        self.fsr = FSRelation(
            in_channels=fpn_max_in_channels,
            channels_list=[fpn_out_channels] * 4,
            out_channels=fsr_out_channels,
            scale_aware_proj=scale_aware_proj)

        self.decoder = AsymmetricDecoder(
            in_channels=fsr_out_channels, out_channels=decoder_out_channels)

        self.cls_head = nn.Sequential(
            DefaultConvBlock(decoder_out_channels, num_classes, 1),
            nn.UpsamplingBilinear2D(scale_factor=4))

    def forward(self, x):
        feature_list = self.encoder(x)

        fpn_feature_list = self.fpn(feature_list)
        scene_feature = self.gap(feature_list[-1])
        refined_feature_list = self.fsr(scene_feature, fpn_feature_list)

        feature = self.decoder(refined_feature_list)
        logit = self.cls_head(feature)
        return [logit]
