#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math
import cv2
import numpy as np
import sys

import paddle
import paddle.nn as nn

from paddlers.models.ppgan.utils.visual import *
from paddlers.models.ppgan.utils.download import get_path_from_url
from paddlers.models.ppgan.models.generators import GFPGANv1Clean
from paddlers.models.ppgan.models.generators import GFPGANv1
from paddlers.models.ppgan.faceutils.face_detection.detection.blazeface.utils import *
GFPGAN_weights = 'https://paddlegan.bj.bcebos.com/models/GFPGAN.pdparams'


class gfp_FaceEnhancement(object):
    def __init__(self, size=512, batch_size=1):
        super(gfp_FaceEnhancement, self).__init__()

        # Initialise the face detector
        model_weights_path = get_path_from_url(GFPGAN_weights)
        model_weights = paddle.load(model_weights_path)

        self.face_enhance = GFPGANv1(out_size=512,
                                     num_style_feat=512,
                                     channel_multiplier=1,
                                     resample_kernel=[1, 3, 3, 1],
                                     decoder_load_path=None,
                                     fix_decoder=True,
                                     num_mlp=8,
                                     lr_mlp=0.01,
                                     input_is_latent=True,
                                     different_w=True,
                                     narrow=1,
                                     sft_half=True)
        self.face_enhance.load_dict(model_weights['net_g_ema'])
        self.face_enhance.eval()
        self.size = size
        self.mask = np.zeros((512, 512), np.float32)
        cv2.rectangle(self.mask, (26, 26), (486, 486), (1, 1, 1), -1,
                      cv2.LINE_AA)
        self.mask = cv2.GaussianBlur(self.mask, (101, 101), 11)
        self.mask = cv2.GaussianBlur(self.mask, (101, 101), 11)
        self.mask = paddle.tile(paddle.to_tensor(
            self.mask).unsqueeze(0).unsqueeze(-1),
                                repeat_times=[batch_size, 1, 1, 3]).numpy()

    def enhance_from_image(self, img):
        if isinstance(img, np.ndarray):
            img, _ = resize_and_crop_image(img, 512)
            img = paddle.to_tensor(img).transpose([2, 0, 1])

        else:
            assert img.shape == [3, 512, 512]
        return self.enhance_from_batch(img.unsqueeze(0))[0]

    def enhance_from_batch(self, img):
        if isinstance(img, np.ndarray):
            img_ori, _ = resize_and_crop_batch(img, 512)
            img = paddle.to_tensor(img_ori).transpose([0, 3, 1, 2])
        else:
            assert img.shape[1:] == [3, 512, 512]
            img_ori = img.transpose([0, 2, 3, 1]).numpy()
        img_t = (img / 255. - 0.5) / 0.5

        with paddle.no_grad():
            out, __ = self.face_enhance(img_t)
        image_tensor = out * 0.5 + 0.5
        image_tensor = image_tensor.transpose([0, 2, 3, 1])  # RGB
        image_numpy = paddle.clip(image_tensor, 0, 1) * 255.0

        out = image_numpy.astype(np.uint8).cpu().numpy()
        return out * self.mask + (1 - self.mask) * img_ori
