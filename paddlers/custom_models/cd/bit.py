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
from paddle.nn.initializer import Normal

from .backbones import resnet
from .layers import Conv3x3, Conv1x1, get_norm_layer, Identity
from .param_init import KaimingInitMixin


class BIT(nn.Layer):
    """
    The BIT implementation based on PaddlePaddle.

    The original article refers to
        H. Chen, et al., "Remote Sensing Image Change Detection With Transformers"
        (https://arxiv.org/abs/2103.00208).

    This implementation adopts pretrained encoders, as opposed to the original work where weights are randomly initialized.

    Args:
        in_channels (int): The number of bands of the input images.
        num_classes (int): The number of target classes.
        backbone (str, optional): The ResNet architecture that is used as the backbone. Currently, only 'resnet18' and 
            'resnet34' are supported. Default: 'resnet18'.
        n_stages (int, optional): The number of ResNet stages used in the backbone, which should be a value in {3,4,5}. 
            Default: 4.
        use_tokenizer (bool, optional): Use a tokenizer or not. Default: True.
        token_len (int, optional): The length of input tokens. Default: 4.
        pool_mode (str, optional): The pooling strategy to obtain input tokens when `use_tokenizer` is set to False. 'max'
            for global max pooling and 'avg' for global average pooling. Default: 'max'.
        pool_size (int, optional): The height and width of the pooled feature maps when `use_tokenizer` is set to False. 
            Default: 2.
        enc_with_pos (bool, optional): Whether to add leanred positional embedding to the input feature sequence of the 
            encoder. Default: True.
        enc_depth (int, optional): The number of attention blocks used in the encoder. Default: 1
        enc_head_dim (int, optional): The embedding dimension of each encoder head. Default: 64.
        dec_depth (int, optional): The number of attention blocks used in the decoder. Default: 8.
        dec_head_dim (int, optional): The embedding dimension of each decoder head. Default: 8.

    Raises:
        ValueError: When an unsupported backbone type is specified, or the number of backbone stages is not 3, 4, or 5.
    """

    def __init__(self,
                 in_channels,
                 num_classes,
                 backbone='resnet18',
                 n_stages=4,
                 use_tokenizer=True,
                 token_len=4,
                 pool_mode='max',
                 pool_size=2,
                 enc_with_pos=True,
                 enc_depth=1,
                 enc_head_dim=64,
                 dec_depth=8,
                 dec_head_dim=8,
                 **backbone_kwargs):
        super(BIT, self).__init__()

        # TODO: reduce hard-coded parameters
        DIM = 32
        MLP_DIM = 2 * DIM
        EBD_DIM = DIM

        self.backbone = Backbone(
            in_channels,
            EBD_DIM,
            arch=backbone,
            n_stages=n_stages,
            **backbone_kwargs)

        self.use_tokenizer = use_tokenizer
        if not use_tokenizer:
            # If a tokenzier is not to be used，then downsample the feature maps.
            self.pool_size = pool_size
            self.pool_mode = pool_mode
            self.token_len = pool_size * pool_size
        else:
            self.conv_att = Conv1x1(32, token_len, bias=False)
            self.token_len = token_len

        self.enc_with_pos = enc_with_pos
        if enc_with_pos:
            self.enc_pos_embedding = self.create_parameter(
                shape=(1, self.token_len * 2, EBD_DIM),
                default_initializer=Normal())

        self.enc_depth = enc_depth
        self.dec_depth = dec_depth
        self.enc_head_dim = enc_head_dim
        self.dec_head_dim = dec_head_dim

        self.encoder = TransformerEncoder(
            dim=DIM,
            depth=enc_depth,
            n_heads=8,
            head_dim=enc_head_dim,
            mlp_dim=MLP_DIM,
            dropout_rate=0.)
        self.decoder = TransformerDecoder(
            dim=DIM,
            depth=dec_depth,
            n_heads=8,
            head_dim=dec_head_dim,
            mlp_dim=MLP_DIM,
            dropout_rate=0.,
            apply_softmax=True)

        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear')
        self.conv_out = nn.Sequential(
            Conv3x3(
                EBD_DIM, EBD_DIM, norm=True, act=True),
            Conv3x3(EBD_DIM, num_classes))

    def _get_semantic_tokens(self, x):
        b, c = x.shape[:2]
        att_map = self.conv_att(x)
        att_map = att_map.reshape((b, self.token_len, 1, -1))
        att_map = F.softmax(att_map, axis=-1)
        x = x.reshape((b, 1, c, -1))
        tokens = (x * att_map).sum(-1)
        return tokens

    def _get_reshaped_tokens(self, x):
        if self.pool_mode == 'max':
            x = F.adaptive_max_pool2d(x, (self.pool_size, self.pool_size))
        elif self.pool_mode == 'avg':
            x = F.adaptive_avg_pool2d(x, (self.pool_size, self.pool_size))
        else:
            x = x
        tokens = x.transpose((0, 2, 3, 1)).flatten(1, 2)
        return tokens

    def encode(self, x):
        if self.enc_with_pos:
            x += self.enc_pos_embedding
        x = self.encoder(x)
        return x

    def decode(self, x, m):
        b, c, h, w = x.shape
        x = x.transpose((0, 2, 3, 1)).flatten(1, 2)
        x = self.decoder(x, m)
        x = x.transpose((0, 2, 1)).reshape((b, c, h, w))
        return x

    def forward(self, t1, t2):
        # Extract features via shared backbone.
        x1 = self.backbone(t1)
        x2 = self.backbone(t2)

        # Tokenization
        if self.use_tokenizer:
            token1 = self._get_semantic_tokens(x1)
            token2 = self._get_semantic_tokens(x2)
        else:
            token1 = self._get_reshaped_tokens(x1)
            token2 = self._get_reshaped_tokens(x2)
            
        # Transformer encoder forward
        token = paddle.concat([token1, token2], axis=1)
        token = self.encode(token)
        token1, token2 = paddle.chunk(token, 2, axis=1)

        # Transformer decoder forward
        y1 = self.decode(x1, token1)
        y2 = self.decode(x2, token2)

        # Feature differencing
        y = paddle.abs(y1 - y2)
        y = self.upsample(y)

        # Classifier forward
        pred = self.conv_out(y)
        return [pred]

    def init_weight(self):
        # Use the default initialization method.
        pass


class Residual(nn.Layer):
    def __init__(self, fn):
        super(Residual, self).__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class Residual2(nn.Layer):
    def __init__(self, fn):
        super(Residual2, self).__init__()
        self.fn = fn

    def forward(self, x1, x2, **kwargs):
        return self.fn(x1, x2, **kwargs) + x1


class PreNorm(nn.Layer):
    def __init__(self, dim, fn):
        super(PreNorm, self).__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class PreNorm2(nn.Layer):
    def __init__(self, dim, fn):
        super(PreNorm2, self).__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x1, x2, **kwargs):
        return self.fn(self.norm(x1), self.norm(x2), **kwargs)


class FeedForward(nn.Sequential):
    def __init__(self, dim, hidden_dim, dropout_rate=0.):
        super(FeedForward, self).__init__(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, dim), nn.Dropout(dropout_rate))


class CrossAttention(nn.Layer):
    def __init__(self,
                 dim,
                 n_heads=8,
                 head_dim=64,
                 dropout_rate=0.,
                 apply_softmax=True):
        super(CrossAttention, self).__init__()

        inner_dim = head_dim * n_heads
        self.n_heads = n_heads
        self.scale = dim**-0.5

        self.apply_softmax = apply_softmax

        self.fc_q = nn.Linear(dim, inner_dim, bias_attr=False)
        self.fc_k = nn.Linear(dim, inner_dim, bias_attr=False)
        self.fc_v = nn.Linear(dim, inner_dim, bias_attr=False)

        self.fc_out = nn.Sequential(
            nn.Linear(inner_dim, dim), nn.Dropout(dropout_rate))

    def forward(self, x, ref):
        b, n = x.shape[:2]
        h = self.n_heads

        q = self.fc_q(x)
        k = self.fc_k(ref)
        v = self.fc_v(ref)

        q = q.reshape((b, n, h, -1)).transpose((0, 2, 1, 3))
        k = k.reshape((b, paddle.shape(ref)[1], h, -1)).transpose((0, 2, 1, 3))
        v = v.reshape((b, paddle.shape(ref)[1], h, -1)).transpose((0, 2, 1, 3))

        mult = paddle.matmul(q, k, transpose_y=True) * self.scale

        if self.apply_softmax:
            mult = F.softmax(mult, axis=-1)

        out = paddle.matmul(mult, v)
        out = out.transpose((0, 2, 1, 3)).flatten(2)
        return self.fc_out(out)


class SelfAttention(CrossAttention):
    def forward(self, x):
        return super(SelfAttention, self).forward(x, x)


class TransformerEncoder(nn.Layer):
    def __init__(self, dim, depth, n_heads, head_dim, mlp_dim, dropout_rate):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.LayerList([])
        for _ in range(depth):
            self.layers.append(
                nn.LayerList([
                    Residual(
                        PreNorm(dim,
                                SelfAttention(dim, n_heads, head_dim,
                                              dropout_rate))),
                    Residual(
                        PreNorm(dim, FeedForward(dim, mlp_dim, dropout_rate)))
                ]))

    def forward(self, x):
        for att, ff in self.layers:
            x = att(x)
            x = ff(x)
        return x


class TransformerDecoder(nn.Layer):
    def __init__(self,
                 dim,
                 depth,
                 n_heads,
                 head_dim,
                 mlp_dim,
                 dropout_rate,
                 apply_softmax=True):
        super(TransformerDecoder, self).__init__()
        self.layers = nn.LayerList([])
        for _ in range(depth):
            self.layers.append(
                nn.LayerList([
                    Residual2(
                        PreNorm2(dim,
                                 CrossAttention(dim, n_heads, head_dim,
                                                dropout_rate, apply_softmax))),
                    Residual(
                        PreNorm(dim, FeedForward(dim, mlp_dim, dropout_rate)))
                ]))

    def forward(self, x, m):
        for att, ff in self.layers:
            x = att(x, m)
            x = ff(x)
        return x


class Backbone(nn.Layer, KaimingInitMixin):
    def __init__(self,
                 in_ch,
                 out_ch=32,
                 arch='resnet18',
                 pretrained=True,
                 n_stages=5):
        super(Backbone, self).__init__()

        expand = 1
        strides = (2, 1, 2, 1, 1)
        if arch == 'resnet18':
            self.resnet = resnet.resnet18(
                pretrained=pretrained,
                strides=strides,
                norm_layer=get_norm_layer())
        elif arch == 'resnet34':
            self.resnet = resnet.resnet34(
                pretrained=pretrained,
                strides=strides,
                norm_layer=get_norm_layer())
        else:
            raise ValueError

        self.n_stages = n_stages

        if self.n_stages == 5:
            itm_ch = 512 * expand
        elif self.n_stages == 4:
            itm_ch = 256 * expand
        elif self.n_stages == 3:
            itm_ch = 128 * expand
        else:
            raise ValueError

        self.upsample = nn.Upsample(scale_factor=2)
        self.conv_out = Conv3x3(itm_ch, out_ch)

        self._trim_resnet()

        if in_ch != 3:
            self.resnet.conv1 = nn.Conv2D(
                in_ch, 64, kernel_size=7, stride=2, padding=3, bias_attr=False)

        if not pretrained:
            self.init_weight()

    def forward(self, x):
        y = self.resnet.conv1(x)
        y = self.resnet.bn1(y)
        y = self.resnet.relu(y)
        y = self.resnet.maxpool(y)

        y = self.resnet.layer1(y)
        y = self.resnet.layer2(y)
        y = self.resnet.layer3(y)
        y = self.resnet.layer4(y)

        y = self.upsample(y)

        return self.conv_out(y)

    def _trim_resnet(self):
        if self.n_stages > 5:
            raise ValueError

        if self.n_stages < 5:
            self.resnet.layer4 = Identity()

        if self.n_stages <= 3:
            self.resnet.layer3 = Identity()

        self.resnet.avgpool = Identity()
        self.resnet.fc = Identity()
