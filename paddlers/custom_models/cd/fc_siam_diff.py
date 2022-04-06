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

# Transferred from https://github.com/rcdaudt/fully_convolutional_change_detection/blob/master/siamunet_diff.py .

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from .layers import Conv3x3, MaxPool2x2, ConvTransposed3x3, Identity


class FCSiamDiff(nn.Layer):
    """
    The FC-Siam-diff implementation based on PaddlePaddle.

    The original article refers to
        Caye Daudt, R., et al. "Fully convolutional siamese networks for change detection"
        (https://arxiv.org/abs/1810.08462).

    Args:
        in_channels (int): The number of bands of the input images.
        num_classes (int): The number of target classes.
        use_dropout (bool, optional): A bool value that indicates whether to use dropout layers. When the model is trained 
            on a relatively small dataset, the dropout layers help prevent overfitting. Default: False.
    """

    def __init__(self, in_channels, num_classes, use_dropout=False):
        super(FCSiamDiff, self).__init__()

        C1, C2, C3, C4, C5 = 16, 32, 64, 128, 256

        self.use_dropout = use_dropout

        self.conv11 = Conv3x3(in_channels, C1, norm=True, act=True)
        self.do11 = self._make_dropout()
        self.conv12 = Conv3x3(C1, C1, norm=True, act=True)
        self.do12 = self._make_dropout()
        self.pool1 = MaxPool2x2()

        self.conv21 = Conv3x3(C1, C2, norm=True, act=True)
        self.do21 = self._make_dropout()
        self.conv22 = Conv3x3(C2, C2, norm=True, act=True)
        self.do22 = self._make_dropout()
        self.pool2 = MaxPool2x2()

        self.conv31 = Conv3x3(C2, C3, norm=True, act=True)
        self.do31 = self._make_dropout()
        self.conv32 = Conv3x3(C3, C3, norm=True, act=True)
        self.do32 = self._make_dropout()
        self.conv33 = Conv3x3(C3, C3, norm=True, act=True)
        self.do33 = self._make_dropout()
        self.pool3 = MaxPool2x2()

        self.conv41 = Conv3x3(C3, C4, norm=True, act=True)
        self.do41 = self._make_dropout()
        self.conv42 = Conv3x3(C4, C4, norm=True, act=True)
        self.do42 = self._make_dropout()
        self.conv43 = Conv3x3(C4, C4, norm=True, act=True)
        self.do43 = self._make_dropout()
        self.pool4 = MaxPool2x2()

        self.upconv4 = ConvTransposed3x3(C4, C4, output_padding=1)

        self.conv43d = Conv3x3(C5, C4, norm=True, act=True)
        self.do43d = self._make_dropout()
        self.conv42d = Conv3x3(C4, C4, norm=True, act=True)
        self.do42d = self._make_dropout()
        self.conv41d = Conv3x3(C4, C3, norm=True, act=True)
        self.do41d = self._make_dropout()

        self.upconv3 = ConvTransposed3x3(C3, C3, output_padding=1)

        self.conv33d = Conv3x3(C4, C3, norm=True, act=True)
        self.do33d = self._make_dropout()
        self.conv32d = Conv3x3(C3, C3, norm=True, act=True)
        self.do32d = self._make_dropout()
        self.conv31d = Conv3x3(C3, C2, norm=True, act=True)
        self.do31d = self._make_dropout()

        self.upconv2 = ConvTransposed3x3(C2, C2, output_padding=1)

        self.conv22d = Conv3x3(C3, C2, norm=True, act=True)
        self.do22d = self._make_dropout()
        self.conv21d = Conv3x3(C2, C1, norm=True, act=True)
        self.do21d = self._make_dropout()

        self.upconv1 = ConvTransposed3x3(C1, C1, output_padding=1)

        self.conv12d = Conv3x3(C2, C1, norm=True, act=True)
        self.do12d = self._make_dropout()
        self.conv11d = Conv3x3(C1, num_classes)

        self.init_weight()

    def forward(self, t1, t2):
        # Encode t1
        # Stage 1
        x11 = self.do11(self.conv11(t1))
        x12_1 = self.do12(self.conv12(x11))
        x1p = self.pool1(x12_1)

        # Stage 2
        x21 = self.do21(self.conv21(x1p))
        x22_1 = self.do22(self.conv22(x21))
        x2p = self.pool2(x22_1)

        # Stage 3
        x31 = self.do31(self.conv31(x2p))
        x32 = self.do32(self.conv32(x31))
        x33_1 = self.do33(self.conv33(x32))
        x3p = self.pool3(x33_1)

        # Stage 4
        x41 = self.do41(self.conv41(x3p))
        x42 = self.do42(self.conv42(x41))
        x43_1 = self.do43(self.conv43(x42))
        x4p = self.pool4(x43_1)

        # Encode t2
        # Stage 1
        x11 = self.do11(self.conv11(t2))
        x12_2 = self.do12(self.conv12(x11))
        x1p = self.pool1(x12_2)

        # Stage 2
        x21 = self.do21(self.conv21(x1p))
        x22_2 = self.do22(self.conv22(x21))
        x2p = self.pool2(x22_2)

        # Stage 3
        x31 = self.do31(self.conv31(x2p))
        x32 = self.do32(self.conv32(x31))
        x33_2 = self.do33(self.conv33(x32))
        x3p = self.pool3(x33_2)

        # Stage 4
        x41 = self.do41(self.conv41(x3p))
        x42 = self.do42(self.conv42(x41))
        x43_2 = self.do43(self.conv43(x42))
        x4p = self.pool4(x43_2)

        # Decode
        # Stage 4d
        x4d = self.upconv4(x4p)
        pad4 = (0, paddle.shape(x43_1)[3] - paddle.shape(x4d)[3], 0,
                paddle.shape(x43_1)[2] - paddle.shape(x4d)[2])
        x4d = F.pad(x4d, pad=pad4, mode='replicate')
        x4d = paddle.concat([x4d, paddle.abs(x43_1 - x43_2)], 1)
        x43d = self.do43d(self.conv43d(x4d))
        x42d = self.do42d(self.conv42d(x43d))
        x41d = self.do41d(self.conv41d(x42d))

        # Stage 3d
        x3d = self.upconv3(x41d)
        pad3 = (0, paddle.shape(x33_1)[3] - paddle.shape(x3d)[3], 0,
                paddle.shape(x33_1)[2] - paddle.shape(x3d)[2])
        x3d = F.pad(x3d, pad=pad3, mode='replicate')
        x3d = paddle.concat([x3d, paddle.abs(x33_1 - x33_2)], 1)
        x33d = self.do33d(self.conv33d(x3d))
        x32d = self.do32d(self.conv32d(x33d))
        x31d = self.do31d(self.conv31d(x32d))

        # Stage 2d
        x2d = self.upconv2(x31d)
        pad2 = (0, paddle.shape(x22_1)[3] - paddle.shape(x2d)[3], 0,
                paddle.shape(x22_1)[2] - paddle.shape(x2d)[2])
        x2d = F.pad(x2d, pad=pad2, mode='replicate')
        x2d = paddle.concat([x2d, paddle.abs(x22_1 - x22_2)], 1)
        x22d = self.do22d(self.conv22d(x2d))
        x21d = self.do21d(self.conv21d(x22d))

        # Stage 1d
        x1d = self.upconv1(x21d)
        pad1 = (0, paddle.shape(x12_1)[3] - paddle.shape(x1d)[3], 0,
                paddle.shape(x12_1)[2] - paddle.shape(x1d)[2])
        x1d = F.pad(x1d, pad=pad1, mode='replicate')
        x1d = paddle.concat([x1d, paddle.abs(x12_1 - x12_2)], 1)
        x12d = self.do12d(self.conv12d(x1d))
        x11d = self.conv11d(x12d)

        return [x11d]

    def init_weight(self):
        pass

    def _make_dropout(self):
        if self.use_dropout:
            return nn.Dropout2D(p=0.2)
        else:
            return Identity()
