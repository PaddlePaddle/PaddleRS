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

import warnings
import math
from functools import partial

import paddle as pd
import paddle.nn as nn
import paddle.nn.functional as F

from .layers.pd_timm import DropPath, to_2tuple


def calc_product(*args):
    if len(args) < 1:
        raise ValueError
    ret = args[0]
    for arg in args[1:]:
        ret *= arg
    return ret


class ConvBlock(pd.nn.Layer):
    def __init__(self,
                 input_size,
                 output_size,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 bias=True,
                 activation='prelu',
                 norm=None):
        super(ConvBlock, self).__init__()
        self.conv = pd.nn.Conv2D(
            input_size,
            output_size,
            kernel_size,
            stride,
            padding,
            bias_attr=bias)

        self.norm = norm
        if self.norm == 'batch':
            self.bn = pd.nn.BatchNorm2D(output_size)
        elif self.norm == 'instance':
            self.bn = pd.nn.InstanceNorm2D(output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = pd.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = pd.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = pd.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = pd.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = pd.nn.Sigmoid()

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.conv(x))
        else:
            out = self.conv(x)

        if self.activation != 'no':
            return self.act(out)
        else:
            return out


class DeconvBlock(pd.nn.Layer):
    def __init__(self,
                 input_size,
                 output_size,
                 kernel_size=4,
                 stride=2,
                 padding=1,
                 bias=True,
                 activation='prelu',
                 norm=None):
        super(DeconvBlock, self).__init__()
        self.deconv = pd.nn.Conv2DTranspose(
            input_size,
            output_size,
            kernel_size,
            stride,
            padding,
            bias_attr=bias)

        self.norm = norm
        if self.norm == 'batch':
            self.bn = pd.nn.BatchNorm2D(output_size)
        elif self.norm == 'instance':
            self.bn = pd.nn.InstanceNorm2D(output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = pd.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = pd.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = pd.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = pd.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = pd.nn.Sigmoid()

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.deconv(x))
        else:
            out = self.deconv(x)

        if self.activation is not None:
            return self.act(out)
        else:
            return out


class ConvLayer(nn.Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvLayer, self).__init__()
        self.conv2d = nn.Conv2D(in_channels, out_channels, kernel_size, stride,
                                padding)

    def forward(self, x):
        out = self.conv2d(x)
        return out


class UpsampleConvLayer(pd.nn.Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(UpsampleConvLayer, self).__init__()
        self.conv2d = nn.Conv2DTranspose(
            in_channels, out_channels, kernel_size, stride=stride, padding=1)

    def forward(self, x):
        out = self.conv2d(x)
        return out


class ResidualBlock(pd.nn.Layer):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(
            channels, channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = ConvLayer(
            channels, channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out) * 0.1
        out = pd.add(out, residual)
        return out


class ChangeFormer(nn.Layer):
    """
    The ChangeFormer implementation based on PaddlePaddle.

    The original article refers to
        Wele Gedara Chaminda Bandara, Vishal M. Patel., "A TRANSFORMER-BASED SIAMESE NETWORK FOR CHANGE DETECTION"
        (https://arxiv.org/pdf/2201.01293.pdf).

    Args:
        in_channels (int): Number of bands of the input images.
        num_classes (int): Number of target classes.
        decoder_softmax (bool, optional): Use softmax after decode or not. Default: False.
        embed_dim (int, optional): Embedding dimension of each decoder head. Default: 256.
    """

    def __init__(self,
                 in_channels,
                 num_classes,
                 decoder_softmax=False,
                 embed_dim=256):
        super(ChangeFormer, self).__init__()

        # Transformer Encoder
        self.embed_dims = [64, 128, 320, 512]
        self.depths = [3, 3, 4, 3]
        self.embedding_dim = embed_dim
        self.drop_rate = 0.1
        self.attn_drop = 0.1
        self.drop_path_rate = 0.1

        self.Tenc_x2 = EncoderTransformer_v3(
            img_size=256,
            patch_size=7,
            in_chans=in_channels,
            num_classes=num_classes,
            embed_dims=self.embed_dims,
            num_heads=[1, 2, 4, 8],
            mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True,
            qk_scale=None,
            drop_rate=self.drop_rate,
            attn_drop_rate=self.attn_drop,
            drop_path_rate=self.drop_path_rate,
            norm_layer=partial(
                nn.LayerNorm, epsilon=1e-6),
            depths=self.depths,
            sr_ratios=[8, 4, 2, 1])

        # Transformer Decoder
        self.TDec_x2 = DecoderTransformer_v3(
            input_transform='multiple_select',
            in_index=[0, 1, 2, 3],
            align_corners=False,
            in_channels=self.embed_dims,
            embedding_dim=self.embedding_dim,
            output_nc=num_classes,
            decoder_softmax=decoder_softmax,
            feature_strides=[2, 4, 8, 16])

    def forward(self, x1, x2):
        [fx1, fx2] = [self.Tenc_x2(x1), self.Tenc_x2(x2)]

        cp = self.TDec_x2(fx1, fx2)

        return [cp]


# Transormer Ecoder with x2, x4, x8, x16 scales
class EncoderTransformer_v3(nn.Layer):
    def __init__(self,
                 img_size=256,
                 patch_size=3,
                 in_chans=3,
                 num_classes=2,
                 embed_dims=[32, 64, 128, 256],
                 num_heads=[2, 2, 4, 8],
                 mlp_ratios=[4, 4, 4, 4],
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=nn.LayerNorm,
                 depths=[3, 3, 6, 18],
                 sr_ratios=[8, 4, 2, 1]):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.embed_dims = embed_dims

        # Patch embedding definitions
        self.patch_embed1 = OverlapPatchEmbed(
            img_size=img_size,
            patch_size=7,
            stride=4,
            in_chans=in_chans,
            embed_dim=embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(
            img_size=img_size // 4,
            patch_size=patch_size,
            stride=2,
            in_chans=embed_dims[0],
            embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(
            img_size=img_size // 8,
            patch_size=patch_size,
            stride=2,
            in_chans=embed_dims[1],
            embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(
            img_size=img_size // 16,
            patch_size=patch_size,
            stride=2,
            in_chans=embed_dims[2],
            embed_dim=embed_dims[3])

        # Stage-1 (x1/4 scale)
        dpr = [x.item() for x in pd.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        self.block1 = nn.LayerList([
            Block(
                dim=embed_dims[0],
                num_heads=num_heads[0],
                mlp_ratio=mlp_ratios[0],
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[cur + i],
                norm_layer=norm_layer,
                sr_ratio=sr_ratios[0]) for i in range(depths[0])
        ])
        self.norm1 = norm_layer(embed_dims[0])

        # Stage-2 (x1/8 scale)
        cur += depths[0]
        self.block2 = nn.LayerList([
            Block(
                dim=embed_dims[1],
                num_heads=num_heads[1],
                mlp_ratio=mlp_ratios[1],
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[cur + i],
                norm_layer=norm_layer,
                sr_ratio=sr_ratios[1]) for i in range(depths[1])
        ])
        self.norm2 = norm_layer(embed_dims[1])

        # Stage-3 (x1/16 scale)
        cur += depths[1]
        self.block3 = nn.LayerList([
            Block(
                dim=embed_dims[2],
                num_heads=num_heads[2],
                mlp_ratio=mlp_ratios[2],
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[cur + i],
                norm_layer=norm_layer,
                sr_ratio=sr_ratios[2]) for i in range(depths[2])
        ])
        self.norm3 = norm_layer(embed_dims[2])

        # Stage-4 (x1/32 scale)
        cur += depths[2]
        self.block4 = nn.LayerList([
            Block(
                dim=embed_dims[3],
                num_heads=num_heads[3],
                mlp_ratio=mlp_ratios[3],
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[cur + i],
                norm_layer=norm_layer,
                sr_ratio=sr_ratios[3]) for i in range(depths[3])
        ])
        self.norm4 = norm_layer(embed_dims[3])

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_op = nn.initializer.TruncatedNormal(std=.02)
            trunc_normal_op(m.weight)

            if isinstance(m, nn.Linear) and m.bias is not None:
                init_bias = nn.initializer.Constant(0)
                init_bias(m.bias)

        elif isinstance(m, nn.LayerNorm):
            init_bias = nn.initializer.Constant(0)
            init_bias(m.bias)

            init_weight = nn.initializer.Constant(1.0)
            init_weight(m.weight)

        elif isinstance(m, nn.Conv2D):
            fan_out = m._kernel_size[0] * m._kernel_size[1] * m._out_channels
            fan_out //= m._groups
            init_weight = nn.initializer.Normal(0, math.sqrt(2.0 / fan_out))
            init_weight(m.weight)

            if m.bias is not None:
                init_bias = nn.initializer.Constant(0)
                init_bias(m.bias)

    def reset_drop_path(self, drop_path_rate):
        dpr = [
            x.item() for x in pd.linspace(0, drop_path_rate, sum(self.depths))
        ]
        cur = 0
        for i in range(self.depths[0]):
            self.block1[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[0]
        for i in range(self.depths[1]):
            self.block2[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[1]
        for i in range(self.depths[2]):
            self.block3[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[2]
        for i in range(self.depths[3]):
            self.block4[i].drop_path.drop_prob = dpr[cur + i]

    def forward_features(self, x):
        B = x.shape[0]
        outs = []

        # Stage 1
        x1, H1, W1 = self.patch_embed1(x)
        for i, blk in enumerate(self.block1):
            x1 = blk(x1, H1, W1)
        x1 = self.norm1(x1)
        x1 = x1.reshape(
            [B, H1, W1, calc_product(*x1.shape[1:]) // (H1 * W1)]).transpose(
                [0, 3, 1, 2])
        outs.append(x1)

        # Stage 2
        x1, H1, W1 = self.patch_embed2(x1)
        for i, blk in enumerate(self.block2):
            x1 = blk(x1, H1, W1)
        x1 = self.norm2(x1)
        x1 = x1.reshape(
            [B, H1, W1, calc_product(*x1.shape[1:]) // (H1 * W1)]).transpose(
                [0, 3, 1, 2])
        outs.append(x1)

        # Stage 3
        x1, H1, W1 = self.patch_embed3(x1)
        for i, blk in enumerate(self.block3):
            x1 = blk(x1, H1, W1)
        x1 = self.norm3(x1)
        x1 = x1.reshape(
            [B, H1, W1, calc_product(*x1.shape[1:]) // (H1 * W1)]).transpose(
                [0, 3, 1, 2])
        outs.append(x1)

        # Stage 4
        x1, H1, W1 = self.patch_embed4(x1)
        for i, blk in enumerate(self.block4):
            x1 = blk(x1, H1, W1)
        x1 = self.norm4(x1)
        x1 = x1.reshape(
            [B, H1, W1, calc_product(*x1.shape[1:]) // (H1 * W1)]).transpose(
                [0, 3, 1, 2])
        outs.append(x1)
        return outs

    def forward(self, x):
        x = self.forward_features(x)
        return x


class DecoderTransformer_v3(nn.Layer):
    """
    Transformer Decoder
    """

    def __init__(self,
                 input_transform='multiple_select',
                 in_index=[0, 1, 2, 3],
                 align_corners=True,
                 in_channels=[32, 64, 128, 256],
                 embedding_dim=64,
                 output_nc=2,
                 decoder_softmax=False,
                 feature_strides=[2, 4, 8, 16]):
        super(DecoderTransformer_v3, self).__init__()

        assert len(feature_strides) == len(in_channels)
        assert min(feature_strides) == feature_strides[0]

        # Settings
        self.feature_strides = feature_strides
        self.input_transform = input_transform
        self.in_index = in_index
        self.align_corners = align_corners
        self.in_channels = in_channels
        self.embedding_dim = embedding_dim
        self.output_nc = output_nc
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        # MLP decoder heads
        self.linear_c4 = MLP(input_dim=c4_in_channels,
                             embed_dim=self.embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels,
                             embed_dim=self.embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels,
                             embed_dim=self.embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels,
                             embed_dim=self.embedding_dim)

        # Convolutional Difference Layers
        self.diff_c4 = conv_diff(
            in_channels=2 * self.embedding_dim, out_channels=self.embedding_dim)
        self.diff_c3 = conv_diff(
            in_channels=2 * self.embedding_dim, out_channels=self.embedding_dim)
        self.diff_c2 = conv_diff(
            in_channels=2 * self.embedding_dim, out_channels=self.embedding_dim)
        self.diff_c1 = conv_diff(
            in_channels=2 * self.embedding_dim, out_channels=self.embedding_dim)

        # Take outputs from middle of the encoder
        self.make_pred_c4 = make_prediction(
            in_channels=self.embedding_dim, out_channels=self.output_nc)
        self.make_pred_c3 = make_prediction(
            in_channels=self.embedding_dim, out_channels=self.output_nc)
        self.make_pred_c2 = make_prediction(
            in_channels=self.embedding_dim, out_channels=self.output_nc)
        self.make_pred_c1 = make_prediction(
            in_channels=self.embedding_dim, out_channels=self.output_nc)

        # Final linear fusion layer
        self.linear_fuse = nn.Sequential(
            nn.Conv2D(
                in_channels=self.embedding_dim * len(in_channels),
                out_channels=self.embedding_dim,
                kernel_size=1),
            nn.BatchNorm2D(self.embedding_dim))

        # Final predction head
        self.convd2x = UpsampleConvLayer(
            self.embedding_dim, self.embedding_dim, kernel_size=4, stride=2)
        self.dense_2x = nn.Sequential(ResidualBlock(self.embedding_dim))
        self.convd1x = UpsampleConvLayer(
            self.embedding_dim, self.embedding_dim, kernel_size=4, stride=2)
        self.dense_1x = nn.Sequential(ResidualBlock(self.embedding_dim))
        self.change_probability = ConvLayer(
            self.embedding_dim,
            self.output_nc,
            kernel_size=3,
            stride=1,
            padding=1)

        # Final activation
        self.output_softmax = decoder_softmax
        self.active = nn.Sigmoid()

    def _transform_inputs(self, inputs):
        """
        Transform inputs for decoder.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
        Returns:
            Tensor: The transformed inputs
        """

        if self.input_transform == 'resize_concat':
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                resize(
                    input=x,
                    size=inputs[0].shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners) for x in inputs
            ]
            inputs = pd.concat(upsampled_inputs, dim=1)
        elif self.input_transform == 'multiple_select':
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs

    def forward(self, inputs1, inputs2):
        # Transforming encoder features (select layers)
        x_1 = self._transform_inputs(inputs1)  # len=4, 1/2, 1/4, 1/8, 1/16
        x_2 = self._transform_inputs(inputs2)  # len=4, 1/2, 1/4, 1/8, 1/16

        # img1 and img2 features
        c1_1, c2_1, c3_1, c4_1 = x_1
        c1_2, c2_2, c3_2, c4_2 = x_2

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4_1.shape

        outputs = []
        # Stage 4: x1/32 scale
        _c4_1 = self.linear_c4(c4_1).transpose([0, 2, 1])
        _c4_1 = _c4_1.reshape([
            n, calc_product(*_c4_1.shape[1:]) //
            (c4_1.shape[2] * c4_1.shape[3]), c4_1.shape[2], c4_1.shape[3]
        ])
        _c4_2 = self.linear_c4(c4_2).transpose([0, 2, 1])
        _c4_2 = _c4_2.reshape([
            n, calc_product(*_c4_2.shape[1:]) //
            (c4_2.shape[2] * c4_2.shape[3]), c4_2.shape[2], c4_2.shape[3]
        ])
        _c4 = self.diff_c4(pd.concat((_c4_1, _c4_2), axis=1))
        p_c4 = self.make_pred_c4(_c4)
        outputs.append(p_c4)
        _c4_up = resize(
            _c4, size=c1_2.shape[2:], mode='bilinear', align_corners=False)

        # Stage 3: x1/16 scale
        _c3_1 = self.linear_c3(c3_1).transpose([0, 2, 1])
        _c3_1 = _c3_1.reshape([
            n, calc_product(*_c3_1.shape[1:]) //
            (c3_1.shape[2] * c3_1.shape[3]), c3_1.shape[2], c3_1.shape[3]
        ])
        _c3_2 = self.linear_c3(c3_2).transpose([0, 2, 1])
        _c3_2 = _c3_2.reshape([
            n, calc_product(*_c3_2.shape[1:]) //
            (c3_2.shape[2] * c3_2.shape[3]), c3_2.shape[2], c3_2.shape[3]
        ])
        _c3 = self.diff_c3(pd.concat((_c3_1, _c3_2), axis=1)) + \
            F.interpolate(_c4, scale_factor=2, mode="bilinear")
        p_c3 = self.make_pred_c3(_c3)
        outputs.append(p_c3)
        _c3_up = resize(
            _c3, size=c1_2.shape[2:], mode='bilinear', align_corners=False)

        # Stage 2: x1/8 scale
        _c2_1 = self.linear_c2(c2_1).transpose([0, 2, 1])
        _c2_1 = _c2_1.reshape([
            n, calc_product(*_c2_1.shape[1:]) //
            (c2_1.shape[2] * c2_1.shape[3]), c2_1.shape[2], c2_1.shape[3]
        ])
        _c2_2 = self.linear_c2(c2_2).transpose([0, 2, 1])
        _c2_2 = _c2_2.reshape([
            n, calc_product(*_c2_2.shape[1:]) //
            (c2_2.shape[2] * c2_2.shape[3]), c2_2.shape[2], c2_2.shape[3]
        ])
        _c2 = self.diff_c2(pd.concat((_c2_1, _c2_2), axis=1)) + \
            F.interpolate(_c3, scale_factor=2, mode="bilinear")
        p_c2 = self.make_pred_c2(_c2)
        outputs.append(p_c2)
        _c2_up = resize(
            _c2, size=c1_2.shape[2:], mode='bilinear', align_corners=False)

        # Stage 1: x1/4 scale
        _c1_1 = self.linear_c1(c1_1).transpose([0, 2, 1])
        _c1_1 = _c1_1.reshape([
            n, calc_product(*_c1_1.shape[1:]) //
            (c1_1.shape[2] * c1_1.shape[3]), c1_1.shape[2], c1_1.shape[3]
        ])
        _c1_2 = self.linear_c1(c1_2).transpose([0, 2, 1])
        _c1_2 = _c1_2.reshape([
            n, calc_product(*_c1_2.shape[1:]) //
            (c1_2.shape[2] * c1_2.shape[3]), c1_2.shape[2], c1_2.shape[3]
        ])
        _c1 = self.diff_c1(pd.concat((_c1_1, _c1_2), axis=1)) + \
            F.interpolate(_c2, scale_factor=2, mode="bilinear")
        p_c1 = self.make_pred_c1(_c1)
        outputs.append(p_c1)

        # Linear Fusion of difference image from all scales
        _c = self.linear_fuse(pd.concat((_c4_up, _c3_up, _c2_up, _c1), axis=1))

        # Upsampling x2 (x1/2 scale)
        x = self.convd2x(_c)
        # Residual block
        x = self.dense_2x(x)
        # Upsampling x2 (x1 scale)
        x = self.convd1x(x)
        # Residual block
        x = self.dense_1x(x)

        # Final prediction
        cp = self.change_probability(x)

        outputs.append(cp)

        if self.output_softmax:
            temp = outputs
            outputs = []
            for pred in temp:
                outputs.append(self.active(pred))

        return outputs[-1]


class OverlapPatchEmbed(nn.Layer):
    """ 
    Image to Patch Embedding
    """

    def __init__(self,
                 img_size=224,
                 patch_size=7,
                 stride=4,
                 in_chans=3,
                 embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[
            1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2D(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_op = nn.initializer.TruncatedNormal(std=.02)
            trunc_normal_op(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                init_bias = nn.initializer.Constant(0)
                init_bias(m.bias)
        elif isinstance(m, nn.LayerNorm):
            init_bias = nn.initializer.Constant(0)
            init_bias(m.bias)
            init_weight = nn.initializer.Constant(1.0)
            init_weight(m.weight)
        elif isinstance(m, nn.Conv2D):
            fan_out = m._kernel_size[0] * m._kernel_size[1] * m._out_channels
            fan_out //= m._groups
            init_weight = nn.initializer.Normal(0, math.sqrt(2.0 / fan_out))
            init_weight(m.weight)
            if m.bias is not None:
                init_bias = nn.initializer.Constant(0)
                init_bias(m.bias)

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose([0, 2, 1])
        x = self.norm(x)

        return x, H, W


def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1 and
                     input_w > 1) and (output_h - 1) % (input_h - 1) and
                    (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    return F.interpolate(input, size, scale_factor, mode, align_corners)


class Mlp(nn.Layer):
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_op = nn.initializer.TruncatedNormal(std=.02)
            trunc_normal_op(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                init_bias = nn.initializer.Constant(0)
                init_bias(m.bias)
        elif isinstance(m, nn.LayerNorm):
            init_bias = nn.initializer.Constant(0)
            init_bias(m.bias)
            init_weight = nn.initializer.Constant(1.0)
            init_weight(m.weight)
        elif isinstance(m, nn.Conv2D):
            fan_out = m._kernel_size[0] * m._kernel_size[1] * m._out_channels
            fan_out //= m._groups
            init_weight = nn.initializer.Normal(0, math.sqrt(2.0 / fan_out))
            init_weight(m.weight)
            if m.bias is not None:
                init_bias = nn.initializer.Constant(0)
                init_bias(m.bias)

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Layer):
    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.q = nn.Linear(dim, dim, bias_attr=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias_attr=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2D(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_op = nn.initializer.TruncatedNormal(std=.02)
            trunc_normal_op(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                init_bias = nn.initializer.Constant(0)
                init_bias(m.bias)
        elif isinstance(m, nn.LayerNorm):
            init_bias = nn.initializer.Constant(0)
            init_bias(m.bias)
            init_weight = nn.initializer.Constant(1.0)
            init_weight(m.weight)
        elif isinstance(m, nn.Conv2D):
            fan_out = m._kernel_size[0] * m._kernel_size[1] * m._out_channels
            fan_out //= m._groups
            init_weight = nn.initializer.Normal(0, math.sqrt(2.0 / fan_out))
            init_weight(m.weight)
            if m.bias is not None:
                init_bias = nn.initializer.Constant(0)
                init_bias(m.bias)

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape([B, N, self.num_heads,
                               C // self.num_heads]).transpose([0, 2, 1, 3])

        if self.sr_ratio > 1:
            x_ = x.transpose([0, 2, 1]).reshape([B, C, H, W])
            x_ = self.sr(x_)
            x_ = x_.reshape([B, C, calc_product(*x_.shape[2:])]).transpose(
                [0, 2, 1])
            x_ = self.norm(x_)
            kv = self.kv(x_)
            kv = kv.reshape([
                B, calc_product(*kv.shape[1:]) // (2 * C), 2, self.num_heads,
                C // self.num_heads
            ]).transpose([2, 0, 3, 1, 4])
        else:
            kv = self.kv(x)
            kv = kv.reshape([
                B, calc_product(*kv.shape[1:]) // (2 * C), 2, self.num_heads,
                C // self.num_heads
            ]).transpose([2, 0, 3, 1, 4])
        k, v = kv[0], kv[1]

        attn = (q @k.transpose([0, 1, 3, 2])) * self.scale
        attn = F.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)

        x = (attn @v).transpose([0, 2, 1, 3]).reshape([B, N, C])
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Layer):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            sr_ratio=sr_ratio)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity(
        )
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim,
                       hidden_features=mlp_hidden_dim,
                       act_layer=act_layer,
                       drop=drop)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_op = nn.initializer.TruncatedNormal(std=.02)
            trunc_normal_op(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                init_bias = nn.initializer.Constant(0)
                init_bias(m.bias)
        elif isinstance(m, nn.LayerNorm):
            init_bias = nn.initializer.Constant(0)
            init_bias(m.bias)
            init_weight = nn.initializer.Constant(1.0)
            init_weight(m.weight)
        elif isinstance(m, nn.Conv2D):
            fan_out = m._kernel_size[0] * m._kernel_size[1] * m._out_channels
            fan_out //= m._groups
            init_weight = nn.initializer.Normal(0, math.sqrt(2.0 / fan_out))
            init_weight(m.weight)
            if m.bias is not None:
                init_bias = nn.initializer.Constant(0)
                init_bias(m.bias)

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x


class DWConv(nn.Layer):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2D(dim, dim, 3, 1, 1, bias_attr=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose([0, 2, 1]).reshape([B, C, H, W])
        x = self.dwconv(x)
        x = x.flatten(2).transpose([0, 2, 1])

        return x


# Transformer Decoder
class MLP(nn.Layer):
    """
    Linear Embedding
    """

    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose([0, 2, 1])
        x = self.proj(x)
        return x


# Difference Layer
def conv_diff(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2D(
            in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.BatchNorm2D(out_channels),
        nn.Conv2D(
            out_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU())


# Intermediate prediction Layer
def make_prediction(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2D(
            in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.BatchNorm2D(out_channels),
        nn.Conv2D(
            out_channels, out_channels, kernel_size=3, padding=1))
