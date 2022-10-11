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

import math
from functools import partial

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.vision.models import resnet

from paddlers.models.ppdet.modeling import initializer as init


def conv_with_kaiming_uniform(use_gn=False, use_relu=False):
    def make_conv(
            in_channels, out_channels, kernel_size, stride=1, dilation=1
    ):
        conv = nn.Conv2D(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=dilation * (kernel_size - 1) // 2,
            dilation=dilation,
            bias_attr=False if use_gn else True
        )

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


class FPN(nn.Layer):

    def __init__(self,
                 in_channels_list,
                 out_channels,
                 conv_block=default_conv_block,
                 top_blocks=None
                 ):

        super(FPN, self).__init__()
        self.inner_blocks = []
        self.layer_blocks = []
        for idx, in_channels in enumerate(in_channels_list, 1):
            inner_block = "fpn_inner{}".format(idx)
            layer_block = "fpn_layer{}".format(idx)

            if in_channels == 0:
                continue
            inner_block_module = conv_block(in_channels, out_channels, 1)
            layer_block_module = conv_block(out_channels, out_channels, 3, 1)
            self.add_sublayer(inner_block, inner_block_module)
            self.add_sublayer(layer_block, layer_block_module)
            self.inner_blocks.append(inner_block)
            self.layer_blocks.append(layer_block)
        self.top_blocks = top_blocks

    def forward(self, x):
        last_inner = getattr(self, self.inner_blocks[-1])(x[-1])
        results = [getattr(self, self.layer_blocks[-1])(last_inner)]
        for i, feature in enumerate(x[-2::-1]):
            inner_block = getattr(self, self.inner_blocks[len(self.inner_blocks) - 2 - i])
            layer_block = getattr(self, self.layer_blocks[len(self.layer_blocks) - 2 - i])
            inner_top_down = F.interpolate(last_inner, scale_factor=2, mode='nearest')
            inner_laternal = inner_block(feature)
            last_inner = inner_laternal + inner_top_down
            results.insert(0, layer_block(last_inner))

        if isinstance(self.top_blocks, LastLevelP6P7):
            last_results = self.top_blocks(x[-1], results[-1])
            results.extend(last_results)
        elif isinstance(self.top_blocks, LastLevelMaxPool):
            last_results = self.top_blocks(results[-1])
            results.extend(last_results)

        return tuple(results)


class LastLevelMaxPool(nn.Layer):
    def forward(self, x):
        return [F.max_pool2d(x, 1, 2, 0)]


class LastLevelP6P7(nn.Layer):

    def __init__(self, in_channels, out_channels):
        super(LastLevelP6P7, self).__init__()
        self.p6 = nn.Conv2D(in_channels, out_channels, 3, 2, 1)
        self.p7 = nn.Conv2D(out_channels, out_channels, 3, 2, 1)
        for module in [self.p6, self.p7]:
            init.kaiming_uniform_(module.weight, a=1)
            init.constant_(module.bias, value=0)
        self.use_P5 = in_channels == out_channels

    def forward(self, c5, p5):
        x = p5 if self.use_P5 else c5
        p6 = self.p6(x)
        p7 = self.p7(F.relu(p6))
        return [p6, p7]


class ResNetEncoder(nn.Layer):

    def __init__(self,
                 backbone,
                 include_conv5,
                 batch_norm_trainable,
                 freeze_at,
                 output_stride,
                 in_channels=3,
                 pretrained=True,
                 with_cp=(False, False, False, False),
                 ):
        super(ResNetEncoder, self).__init__()
        if all([output_stride != 16,
                output_stride != 32,
                output_stride != 8]):
            raise ValueError('output_stride must be 8, 16 or 32.')
        self.with_cp = with_cp
        self.freeze_at = freeze_at
        self.batch_norm_trainable = batch_norm_trainable
        self.include_conv5 = include_conv5
        self.resnet = getattr(resnet, backbone)(pretrained=pretrained)
        if in_channels != 3:
            self.resnet.conv1 = nn.Conv2D(
                in_channels, 64, 7, stride=2, padding=3, bias_attr=False)

        print('ResNetEncoder: pretrained = {}'.format(pretrained))
        self.resnet._sub_layers.pop('fc')

        if output_stride == 16:
            self.resnet.layer4.apply(partial(self._nostride_dilate, dilate=2))
        elif output_stride == 8:
            self.resnet.layer3.apply(partial(self._nostride_dilate, dilate=2))
            self.resnet.layer4.apply(partial(self._nostride_dilate, dilate=4))

    @property
    def layer1(self):
        return self.resnet.layer1

    @layer1.setter
    def layer1(self, value):
        del self.resnet.layer1
        self.resnet.layer1 = value

    @property
    def layer2(self):
        return self.resnet.layer2

    @layer2.setter
    def layer2(self, value):
        del self.resnet.layer2
        self.resnet.layer2 = value

    @property
    def layer3(self):
        return self.resnet.layer3

    @layer3.setter
    def layer3(self, value):
        del self.resnet.layer3
        self.resnet.layer3 = value

    @property
    def layer4(self):
        return self.resnet.layer4

    @layer4.setter
    def layer4(self, value):
        del self.resnet.layer4
        self.resnet.layer4 = value

    @staticmethod
    def get_function(module):
        def _function(x):
            y = module(x)
            return y

        return _function

    def forward(self, inputs):
        c2, c3, c4 = None, None, None
        x = inputs
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        if self.with_cp[0] and not x.stop_gradient:
            pass
        else:
            c2 = self.resnet.layer1(x)

        if self.with_cp[1] and not c2.stop_gradient:
            pass
        else:
            c3 = self.resnet.layer2(c2)

        if self.with_cp[2] and not c3.stop_gradient:
            pass
        else:
            c4 = self.resnet.layer3(c3)

        if self.include_conv5:
            c5 = self.resnet.layer4(c4)
            return [c2, c3, c4, c5]

        return [c2, c3, c4]

    def train(self, mode=True):
        super(ResNetEncoder, self).train()
        self._freeze_at(self.freeze_at)
        if mode and not self.batchnorm_trainable:
            for m in self.modules():
                if isinstance(m, nn.LayerList.batchnorm._BatchNorm):
                    m.eval()

    def _nostride_dilate(self, m, dilate):
        class_name = m.__class__.__name__
        if class_name.find('Conv') != -1:

            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate // 2, dilate // 2)
                    m.padding = (dilate // 2, dilate // 2)
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)


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
                raise ValueError('When norm_fn is nn.GroupNorm, num_groups_gn is needed.')
            norm_fn_args = dict(num_groups=num_groups_gn, num_channels=out_channels)
        else:
            raise ValueError('Type of {} is not support.'.format(type(norm_fn)))
        self.blocks = nn.LayerList()
        for in_feat_os in in_feat_output_strides:
            num_upsample = int(math.log2(int(in_feat_os))) - int(math.log2(int(out_feat_output_stride)))

            num_layers = num_upsample if num_upsample != 0 else 1

            self.blocks.append(nn.Sequential(*[
                nn.Sequential(
                    nn.Conv2D(in_channels if idx == 0 else out_channels, out_channels, 3, 1, 1, bias_attr=False),
                    norm_fn(**norm_fn_args) if norm_fn is not None else nn.Identity(),
                    nn.ReLU(),
                    nn.UpsamplingBilinear2D(scale_factor=2) if num_upsample != 0 else nn.Identity(),
                )
                for idx in range(num_layers)]))

    def forward(self, feat_list: list):
        inner_feat_list = []
        for idx, block in enumerate(self.blocks):
            decoder_feat = block(feat_list[idx])
            inner_feat_list.append(decoder_feat)

        out_feat = sum(inner_feat_list) / 4.
        return out_feat


def som(loss, ratio):
    num_inst = loss.numel()
    num_hns = int(ratio * num_inst)

    top_loss, _ = loss.reshape(-1).topk(num_hns, -1)
    loss_mask = (top_loss != 0)

    return paddle.sum(top_loss[loss_mask]) / (loss_mask.sum())


class FactSeg(nn.Layer):
    """
     The FactSeg implementation based on PaddlePaddle.

     The original article refers to
     A. Ma, J. Wang, Y. Zhong and Z. Zheng, "FactSeg: Foreground Activation-Driven Small Object Semantic Segmentation
     in Large-Scale Remote Sensing Imagery," in IEEE Transactions on Geoscience and Remote Sensing, vol. 60,
     pp. 1-16, 2022, Art no. 5606216.


     Args:
         in_channels (int): The number of image channels for the input model. Default: 3.
         num_classes (int): The unique number of target classes. Default: 16.
         backbone (str): A backbone network, models available in `paddle.vision.models.resnet`. Default: resnet50.
         backbone_pretrained (bool): Whether the backbone network uses IMAGENET pretrained weights. Default: True.
     """

    def __init__(self,
                 in_channels=3,
                 num_classes=16,
                 backbone='resnet50',
                 backbone_pretrained=True,
                 ):
        super(FactSeg, self).__init__()

        self.resencoder = ResNetEncoder(
            backbone=backbone,
            include_conv5=True,
            batch_norm_trainable=True,
            pretrained=backbone_pretrained,
            freeze_at=0,
            output_stride=32,
            in_channels=in_channels,
            with_cp=(False, False, False, False))

        print('use fpn!')
        self.fgfpn = FPN(in_channels_list=[256, 512, 1024, 2048],
                         out_channels=256)
        self.bifpn = FPN(in_channels_list=[256, 512, 1024, 2048],
                         out_channels=256, )

        self.fg_decoder = AssymetricDecoder(in_channels=256,
                                            out_channels=128,
                                            in_feat_output_strides=(4, 8, 16, 32),
                                            out_feat_output_stride=4)

        self.bi_decoder = AssymetricDecoder(in_channels=256,
                                            out_channels=128,
                                            in_feat_output_strides=(4, 8, 16, 32),
                                            out_feat_output_stride=4)

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
                    forefeat_list[i] = self.fbatt_block_list[i](binaryfeat_list[i], forefeat_list[i])

            fg_out = self.fg_decoder(forefeat_list)
            bi_out = self.bi_decoder(binaryfeat_list)

        fg_pred = self.fg_cls(fg_out)
        bi_pred = self.bi_cls(bi_out)
        fg_pred = F.interpolate(fg_pred, scale_factor=4.0, mode='bilinear',
                                align_corners=True)
        bi_pred = F.interpolate(bi_pred, scale_factor=4.0, mode='bilinear',
                                align_corners=True)

        if self.training:
            return [fg_pred]

        else:
            binary_prob = F.sigmoid(bi_pred)
            cls_prob = F.softmax(fg_pred, axis=1)
            cls_prob[:, 0, :, :] = cls_prob[:, 0, :, :] * (1 - binary_prob).squeeze(axis=1)
            cls_prob[:, 1:, :, :] = cls_prob[:, 1:, :, :] * binary_prob
            z = paddle.sum(cls_prob, axis=1)
            cls_prob = paddle.divide(cls_prob, z)
            return [cls_prob]
