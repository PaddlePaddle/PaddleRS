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
"""
This code is based on https://github.com/Z-Zheng/FarSeg
Ths copyright of Z-Zheng/FarSeg is as follows:
Apache License [see LICENSE for details]
"""

import math
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.vision.models import resnet50
from .fpn import FPN
from ..utils import Identity


class SceneRelation(nn.Layer):
    def __init__(self,
                 in_channels,
                 channel_list,
                 out_channels,
                 scale_aware_proj=True):
        super(SceneRelation, self).__init__()
        self.scale_aware_proj = scale_aware_proj
        if scale_aware_proj:
            self.scene_encoder = nn.LayerList([
                nn.Sequential(
                    nn.Conv2D(in_channels, out_channels, 1),
                    nn.ReLU(), nn.Conv2D(out_channels, out_channels, 1))
                for _ in range(len(channel_list))
            ])
        else:
            # 2mlp
            self.scene_encoder = nn.Sequential(
                nn.Conv2D(in_channels, out_channels, 1),
                nn.ReLU(),
                nn.Conv2D(out_channels, out_channels, 1), )
        self.content_encoders = nn.LayerList()
        self.feature_reencoders = nn.LayerList()
        for c in channel_list:
            self.content_encoders.append(
                nn.Sequential(
                    nn.Conv2D(c, out_channels, 1),
                    nn.BatchNorm2D(out_channels), nn.ReLU()))
            self.feature_reencoders.append(
                nn.Sequential(
                    nn.Conv2D(c, out_channels, 1),
                    nn.BatchNorm2D(out_channels), nn.ReLU()))
        self.normalizer = nn.Sigmoid()

    def forward(self, scene_feature, features: list):
        content_feats = [
            c_en(p_feat)
            for c_en, p_feat in zip(self.content_encoders, features)
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
            op(p_feat) for op, p_feat in zip(self.feature_reencoders, features)
        ]
        refined_feats = [r * p for r, p in zip(relations, p_feats)]
        return refined_feats


class AssymetricDecoder(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 in_feat_output_strides=(4, 8, 16, 32),
                 out_feat_output_stride=4,
                 norm_fn=nn.BatchNorm2D,
                 num_groups_gn=None):
        super(AssymetricDecoder, self).__init__()
        if norm_fn == nn.BatchNorm2D:
            norm_fn_args = dict(num_features=out_channels)
        elif norm_fn == nn.GroupNorm:
            if num_groups_gn is None:
                raise ValueError(
                    'When norm_fn is nn.GroupNorm, num_groups_gn is needed.')
            norm_fn_args = dict(
                num_groups=num_groups_gn, num_channels=out_channels)
        else:
            raise ValueError('Type of {} is not support.'.format(type(norm_fn)))
        self.blocks = nn.LayerList()
        for in_feat_os in in_feat_output_strides:
            num_upsample = int(math.log2(int(in_feat_os))) - int(
                math.log2(int(out_feat_output_stride)))
            num_layers = num_upsample if num_upsample != 0 else 1
            self.blocks.append(
                nn.Sequential(*[
                    nn.Sequential(
                        nn.Conv2D(
                            in_channels if idx == 0 else out_channels,
                            out_channels,
                            3,
                            1,
                            1,
                            bias_attr=False),
                        norm_fn(**norm_fn_args)
                        if norm_fn is not None else Identity(),
                        nn.ReLU(),
                        nn.UpsamplingBilinear2D(scale_factor=2) if num_upsample
                        != 0 else Identity(), ) for idx in range(num_layers)
                ]))

    def forward(self, feat_list: list):
        inner_feat_list = []
        for idx, block in enumerate(self.blocks):
            decoder_feat = block(feat_list[idx])
            inner_feat_list.append(decoder_feat)
        out_feat = sum(inner_feat_list) / 4.
        return out_feat


class ResNet50Encoder(nn.Layer):
    def __init__(self, pretrained=True):
        super(ResNet50Encoder, self).__init__()
        self.resnet = resnet50(pretrained=pretrained)

    def forward(self, inputs):
        x = inputs
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        c2 = self.resnet.layer1(x)
        c3 = self.resnet.layer2(c2)
        c4 = self.resnet.layer3(c3)
        c5 = self.resnet.layer4(c4)
        return [c2, c3, c4, c5]


class FarSeg(nn.Layer):
    '''
        The FarSeg implementation based on PaddlePaddle.

        The original article refers to
        Zheng, Zhuo, et al. "Foreground-Aware Relation Network for Geospatial Object Segmentation in High Spatial Resolution Remote Sensing Imagery"
        (https://openaccess.thecvf.com/content_CVPR_2020/papers/Zheng_Foreground-Aware_Relation_Network_for_Geospatial_Object_Segmentation_in_High_Spatial_CVPR_2020_paper.pdf)
    '''

    def __init__(self,
                 num_classes=16,
                 fpn_ch_list=(256, 512, 1024, 2048),
                 mid_ch=256,
                 out_ch=128,
                 sr_ch_list=(256, 256, 256, 256),
                 encoder_pretrained=True):
        super(FarSeg, self).__init__()
        self.en = ResNet50Encoder(encoder_pretrained)
        self.fpn = FPN(in_channels_list=fpn_ch_list, out_channels=mid_ch)
        self.decoder = AssymetricDecoder(
            in_channels=mid_ch, out_channels=out_ch)
        self.cls_pred_conv = nn.Conv2D(out_ch, num_classes, 1)
        self.upsample4x_op = nn.UpsamplingBilinear2D(scale_factor=4)
        self.scene_relation = True if sr_ch_list is not None else False
        if self.scene_relation:
            self.gap = nn.AdaptiveAvgPool2D(1)
            self.sr = SceneRelation(fpn_ch_list[-1], sr_ch_list, mid_ch)

    def forward(self, x):
        feat_list = self.en(x)
        fpn_feat_list = self.fpn(feat_list)
        if self.scene_relation:
            c5 = feat_list[-1]
            c6 = self.gap(c5)
            refined_fpn_feat_list = self.sr(c6, fpn_feat_list)
        else:
            refined_fpn_feat_list = fpn_feat_list
        final_feat = self.decoder(refined_fpn_feat_list)
        cls_pred = self.cls_pred_conv(final_feat)
        cls_pred = self.upsample4x_op(cls_pred)
        cls_pred = F.softmax(cls_pred, axis=1)
        return [cls_pred]
