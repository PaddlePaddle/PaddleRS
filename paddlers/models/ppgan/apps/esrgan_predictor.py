#  Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

import os
import cv2
import glob
import numpy as np
from PIL import Image
from tqdm import tqdm

import paddle
from ..models.generators import RRDBNet
from ..utils.download import get_path_from_url
from ..utils.logger import get_logger

from .base_predictor import BasePredictor

SR_WEIGHT_URL = 'https://paddlegan.bj.bcebos.com/models/esrgan_x4.pdparams'

class ESRGANPredictor(BasePredictor):
    def __init__(self, output='output', weight_path=None):
        self.input = input
        self.output = os.path.join(output, 'ESRGAN')
        self.model = RRDBNet(3, 3, 64, 23)
        if weight_path is None:
            weight_path = get_path_from_url(SR_WEIGHT_URL)

        state_dict = paddle.load(weight_path)
        state_dict = state_dict['generator']
        self.model.load_dict(state_dict)
        self.model.eval()

    def norm(self, img):
        img = np.array(img).transpose([2, 0, 1]).astype('float32') / 255.0
        return img.astype('float32')

    def denorm(self, img):
        img = img.transpose((1, 2, 0))
        return (img * 255).clip(0, 255).astype('uint8')

    def run_image(self, img):
        if isinstance(img, str):
            ori_img = Image.open(img).convert('RGB')
        elif isinstance(img, np.ndarray):
            ori_img = Image.fromarray(img).convert('RGB')
        elif isinstance(img, Image.Image):
            ori_img = img

        img = self.norm(ori_img)
        x = paddle.to_tensor(img[np.newaxis, ...])
        with paddle.no_grad():
            out = self.model(x)

        pred_img = self.denorm(out.numpy()[0])
        pred_img = Image.fromarray(pred_img)
        return pred_img

    def run(self, input):
        if not os.path.exists(self.output):
            os.makedirs(self.output)

        pred_img = self.run_image(input)

        out_path = None
        if self.output:
            try:
                base_name = os.path.splitext(os.path.basename(input))[0]
            except:
                base_name = 'result'
            out_path = os.path.join(self.output, base_name + '.png')
            pred_img.save(out_path)
            logger = get_logger()
            logger.info('Image saved to {}'.format(out_path))

        return pred_img, out_path