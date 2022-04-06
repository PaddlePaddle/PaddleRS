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

from paddlers.datasets.cd_dataset import MaskType
from paddlers.custom_models.seg import FarSeg
from .layers import Conv3x3, Identity


class _ChangeStarBase(nn.Layer):

    USE_MULTITASK_DECODER = True
    OUT_TYPES = (MaskType.CD, MaskType.CD, MaskType.SEG_T1, MaskType.SEG_T2)

    def __init__(self, seg_model, num_classes, mid_channels, inner_channels,
                 num_convs, scale_factor):
        super(_ChangeStarBase, self).__init__(_ChangeStarBase, self)

        self.extract = seg_model
        self.detect = ChangeMixin(
            in_ch=mid_channels * 2,
            out_ch=num_classes,
            mid_ch=inner_channels,
            num_convs=num_convs,
            scale_factor=scale_factor)
        self.segment = nn.Sequential(
            Conv3x3(mid_channels, 2),
            nn.UpsamplingBilinear2D(scale_factor=scale_factor))

        self.init_weight()

    def forward(self, t1, t2):
        x1 = self.extract(t1)[0]
        x2 = self.extract(t2)[0]
        logit12, logit21 = self.detect(x1, x2)

        if not self.training:
            logit_list = [logit12]
        else:
            logit1 = self.segment(x1)
            logit2 = self.segment(x2)
            logit_list = [logit12, logit21, logit1, logit2]

        return logit_list

    def init_weight(self):
        pass


class ChangeMixin(nn.Layer):
    def __init__(self, in_ch, out_ch, mid_ch, num_convs, scale_factor):
        super(ChangeMixin, self).__init__(ChangeMixin, self)
        convs = [Conv3x3(in_ch, mid_ch, norm=True, act=True)]
        convs += [
            Conv3x3(
                mid_ch, mid_ch, norm=True, act=True)
            for _ in range(num_convs - 1)
        ]
        self.detect = nn.Sequential(
            *convs,
            Conv3x3(mid_ch, out_ch),
            nn.UpsamplingBilinear2D(scale_factor=scale_factor))

    def forward(self, x1, x2):
        pred12 = self.detect(paddle.concat([x1, x2], axis=1))
        pred21 = self.detect(paddle.concat([x2, x1], axis=1))
        return pred12, pred21


class ChangeStar_FarSeg(_ChangeStarBase):
    """
    The ChangeStar implementation with a FarSeg encoder based on PaddlePaddle.

    The original article refers to
        Z. Zheng, et al., "Change is Everywhere: Single-Temporal Supervised Object Change Detection in Remote Sensing Imagery"
        (https://arxiv.org/abs/2108.07002).
    
    Note that this implementation differs from the original code in two aspects:
    1. The encoder of the FarSeg model is ResNet50.
    2. We use conv-bn-relu instead of conv-relu-bn.

    Args:
        num_classes (int): The number of target classes.
        mid_channels (int, optional): The number of channels required by the ChangeMixin module. Default: 256.
        inner_channels (int, optional): The number of filters used in the convolutional layers in the ChangeMixin module. 
            Default: 16.
        num_convs (int, optional): The number of convolutional layers used in the ChangeMixin module. Default: 4.
        scale_factor (float, optional): The scaling factor of the output upsampling layer. Default: 4.0.
    """

    def __init__(
            self,
            num_classes,
            mid_channels=256,
            inner_channels=16,
            num_convs=4,
            scale_factor=4.0, ):
        # TODO: Configurable FarSeg model
        class _FarSegWrapper(nn.Layer):
            def __init__(self, seg_model):
                super(_FarSegWrapper, self).__init__()
                self._seg_model = seg_model
                self._seg_model.cls_pred_conv = Identity()

            def forward(self, x):
                feat_list = self._seg_model.en(x)
                fpn_feat_list = self._seg_model.fpn(feat_list)
                if self._seg_model.scene_relation:
                    c5 = feat_list[-1]
                    c6 = self._seg_model.gap(c5)
                    refined_fpn_feat_list = self._seg_model.sr(c6,
                                                               fpn_feat_list)
                else:
                    refined_fpn_feat_list = fpn_feat_list
                final_feat = self._seg_model.decoder(refined_fpn_feat_list)
                return [final_feat]

        seg_model = FarSeg(out_ch=mid_channels)

        super(ChangeStar_FarSeg, self).__init__(
            seg_model=_FarSegWrapper(seg_model),
            num_classes=num_classes,
            mid_channels=mid_channels,
            inner_channels=inner_channels,
            num_convs=num_convs,
            scale_factor=scale_factor)


# NOTE: Currently, ChangeStar = FarSeg + ChangeMixin + SegHead
ChangeStar = ChangeStar_FarSeg
