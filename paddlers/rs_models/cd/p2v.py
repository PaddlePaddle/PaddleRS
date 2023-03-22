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

from .layers import Conv1x1, Conv3x3, MaxPool2x2


class SimpleResBlock(nn.Layer):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = Conv3x3(in_ch, out_ch, norm=True, act=True)
        self.conv2 = Conv3x3(out_ch, out_ch, norm=True)

    def forward(self, x):
        x = self.conv1(x)
        return F.relu(x + self.conv2(x))


class ResBlock(nn.Layer):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = Conv3x3(in_ch, out_ch, norm=True, act=True)
        self.conv2 = Conv3x3(out_ch, out_ch, norm=True, act=True)
        self.conv3 = Conv3x3(out_ch, out_ch, norm=True)

    def forward(self, x):
        x = self.conv1(x)
        return F.relu(x + self.conv3(self.conv2(x)))


class DecBlock(nn.Layer):
    def __init__(self, in_ch1, in_ch2, out_ch):
        super().__init__()
        self.conv_fuse = SimpleResBlock(in_ch1 + in_ch2, out_ch)

    def forward(self, x1, x2):
        x2 = F.interpolate(x2, size=x1.shape[2:])
        x = paddle.concat([x1, x2], axis=1)
        return self.conv_fuse(x)


class BasicConv3D(nn.Layer):
    def __init__(self,
                 in_ch,
                 out_ch,
                 kernel_size,
                 bias='auto',
                 bn=False,
                 act=False,
                 **kwargs):
        super().__init__()
        seq = []
        if kernel_size >= 2:
            seq.append(nn.Pad3D(kernel_size // 2, mode='constant'))
        seq.append(
            nn.Conv3D(
                in_ch,
                out_ch,
                kernel_size,
                padding=0,
                bias_attr=(False if bn else None) if bias == 'auto' else bias,
                **kwargs))
        if bn:
            seq.append(nn.BatchNorm3D(out_ch))
        if act:
            seq.append(nn.ReLU())
        self.seq = nn.Sequential(*seq)

    def forward(self, x):
        return self.seq(x)


class Conv3x3x3(BasicConv3D):
    def __init__(self,
                 in_ch,
                 out_ch,
                 bias='auto',
                 bn=False,
                 act=False,
                 **kwargs):
        super().__init__(in_ch, out_ch, 3, bias=bias, bn=bn, act=act, **kwargs)


class ResBlock3D(nn.Layer):
    def __init__(self, in_ch, out_ch, itm_ch, stride=1, ds=None):
        super().__init__()
        self.conv1 = BasicConv3D(
            in_ch, itm_ch, 1, bn=True, act=True, stride=stride)
        self.conv2 = Conv3x3x3(itm_ch, itm_ch, bn=True, act=True)
        self.conv3 = BasicConv3D(itm_ch, out_ch, 1, bn=True, act=False)
        self.ds = ds

    def forward(self, x):
        res = x
        y = self.conv1(x)
        y = self.conv2(y)
        y = self.conv3(y)
        if self.ds is not None:
            res = self.ds(res)
        y = F.relu(y + res)
        return y


class PairEncoder(nn.Layer):
    def __init__(self, in_ch, enc_chs=(16, 32, 64), add_chs=(0, 0)):
        super().__init__()

        self.n_layers = 3

        self.conv1 = SimpleResBlock(2 * in_ch, enc_chs[0])
        self.pool1 = MaxPool2x2()

        self.conv2 = SimpleResBlock(enc_chs[0] + add_chs[0], enc_chs[1])
        self.pool2 = MaxPool2x2()

        self.conv3 = ResBlock(enc_chs[1] + add_chs[1], enc_chs[2])
        self.pool3 = MaxPool2x2()

    def forward(self, x1, x2, add_feats=None):
        x = paddle.concat([x1, x2], axis=1)
        feats = [x]

        for i in range(self.n_layers):
            conv = getattr(self, f'conv{i+1}')
            if i > 0 and add_feats is not None:
                add_feat = F.interpolate(add_feats[i - 1], size=x.shape[2:])
                x = paddle.concat([x, add_feat], axis=1)
            x = conv(x)
            pool = getattr(self, f'pool{i+1}')
            x = pool(x)
            feats.append(x)

        return feats


class VideoEncoder(nn.Layer):
    def __init__(self, in_ch, enc_chs=(64, 128)):
        super().__init__()
        if in_ch != 3:
            raise NotImplementedError

        self.n_layers = 2
        self.expansion = 4
        self.tem_scales = (1.0, 0.5)

        self.stem = nn.Sequential(
            nn.Conv3D(
                3,
                enc_chs[0],
                kernel_size=(3, 9, 9),
                stride=(1, 4, 4),
                padding=(1, 4, 4),
                bias_attr=False),
            nn.BatchNorm3D(enc_chs[0]),
            nn.ReLU())
        exps = self.expansion
        self.layer1 = nn.Sequential(
            ResBlock3D(
                enc_chs[0],
                enc_chs[0] * exps,
                enc_chs[0],
                ds=BasicConv3D(
                    enc_chs[0], enc_chs[0] * exps, 1, bn=True)),
            ResBlock3D(enc_chs[0] * exps, enc_chs[0] * exps, enc_chs[0]))
        self.layer2 = nn.Sequential(
            ResBlock3D(
                enc_chs[0] * exps,
                enc_chs[1] * exps,
                enc_chs[1],
                stride=(2, 2, 2),
                ds=BasicConv3D(
                    enc_chs[0] * exps,
                    enc_chs[1] * exps,
                    1,
                    stride=(2, 2, 2),
                    bn=True)),
            ResBlock3D(enc_chs[1] * exps, enc_chs[1] * exps, enc_chs[1]))

    def forward(self, x):
        feats = [x]

        x = self.stem(x)
        for i in range(self.n_layers):
            layer = getattr(self, f'layer{i+1}')
            x = layer(x)
            feats.append(x)

        return feats


class SimpleDecoder(nn.Layer):
    def __init__(self, itm_ch, enc_chs, dec_chs, num_classes=1):
        super().__init__()

        enc_chs = enc_chs[::-1]
        self.conv_bottom = Conv3x3(itm_ch, itm_ch, norm=True, act=True)
        self.blocks = nn.LayerList([
            DecBlock(in_ch1, in_ch2, out_ch)
            for in_ch1, in_ch2, out_ch in zip(enc_chs, (itm_ch, ) +
                                              dec_chs[:-1], dec_chs)
        ])
        self.conv_out = Conv1x1(dec_chs[-1], num_classes)

    def forward(self, x, feats):
        feats = feats[::-1]

        x = self.conv_bottom(x)

        for feat, blk in zip(feats, self.blocks):
            x = blk(feat, x)

        y = self.conv_out(x)

        return y


class P2V(nn.Layer):
    """
    The P2V-CD implementation based on PaddlePaddle.

    The original article refers to
        M. Lin, et al. "Transition Is a Process: Pair-to-Video Change Detection Networks 
        for Very High Resolution Remote Sensing Images"
        (https://ieeexplore.ieee.org/document/9975266).

    Args:
        in_channels (int): Number of bands of the input images.
        num_classes (int): Number of target classes.
        video_len (int, optional): Number of frames of the constructed pseudo video. 
            Default: 8.
        pair_encoder_channels (tuple[int], optional): Output channels of each block in the 
            spatial (pair) encoder. Default: (32, 64, 128).
        video_encoder_channels (tuple[int], optional): Output channels of each block in the
            temporal (video) encoder. Default: (64, 128).
        decoder_channels (tuple[int], optional): Output channels of each block in the 
            decoder. Default: (256, 128, 64, 32).
    """

    def __init__(self,
                 in_channels,
                 num_classes,
                 video_len=8,
                 pair_encoder_channels=(32, 64, 128),
                 video_encoder_channels=(64, 128),
                 decoder_channels=(256, 128, 64, 32)):
        super().__init__()
        if video_len < 2:
            raise ValueError
        self.video_len = video_len
        self.encoder_v = VideoEncoder(
            in_channels, enc_chs=video_encoder_channels)
        video_encoder_channels = tuple(ch * self.encoder_v.expansion
                                       for ch in video_encoder_channels)
        self.encoder_p = PairEncoder(
            in_channels,
            enc_chs=pair_encoder_channels,
            add_chs=video_encoder_channels)
        self.conv_out_v = Conv1x1(video_encoder_channels[-1], num_classes)
        self.convs_video = nn.LayerList([
            Conv1x1(
                2 * ch, ch, norm=True, act=True)
            for ch in video_encoder_channels
        ])
        self.decoder = SimpleDecoder(
            pair_encoder_channels[-1],
            (2 * in_channels, ) + pair_encoder_channels, decoder_channels,
            num_classes)

    def forward(self, t1, t2):
        frames = self.pair_to_video(t1, t2)
        feats_v = self.encoder_v(frames.transpose((0, 2, 1, 3, 4)))
        feats_v.pop(0)

        for i, feat in enumerate(feats_v):
            feats_v[i] = self.convs_video[i](self.tem_aggr(feat))

        feats_p = self.encoder_p(t1, t2, feats_v)

        pred = self.decoder(feats_p[-1], feats_p)

        if self.training:
            pred_v = self.conv_out_v(feats_v[-1])
            pred_v = F.interpolate(pred_v, size=pred.shape[2:])
            return [pred, pred_v]
        else:
            return [pred]

    def pair_to_video(self, im1, im2, rate_map=None):
        def _interpolate(im1, im2, rate_map, len):
            delta = 1.0 / (len - 1)
            delta_map = rate_map * delta
            steps = paddle.arange(
                end=len, dtype='float32').reshape((1, -1, 1, 1, 1))
            interped = im1.unsqueeze(1) + (
                (im2 - im1) * delta_map).unsqueeze(1) * steps
            return interped

        if rate_map is None:
            rate_map = paddle.ones_like(im1[:, 0:1])
        frames = _interpolate(im1, im2, rate_map, self.video_len)
        return frames

    def tem_aggr(self, f):
        return paddle.concat(
            [paddle.mean(
                f, axis=2), paddle.max(f, axis=2)], axis=1)
