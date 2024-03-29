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

import cv2
from glob import glob
from natsort import natsorted
import numpy as np
import os
import random
from tqdm import tqdm

import paddle

from paddlers.models.ppgan.models.generators import SwinIR
from paddlers.models.ppgan.utils.download import get_path_from_url
from .base_predictor import BasePredictor

model_cfgs = {
    'Denoising': {
        'model_urls':
        'https://paddlegan.bj.bcebos.com/models/SwinIR_Denoising.pdparams',
        'upscale': 1,
        'img_size': 128,
        'window_size': 8,
        'depths': [6, 6, 6, 6, 6, 6],
        'embed_dim': 180,
        'num_heads': [6, 6, 6, 6, 6, 6],
        'mlp_ratio': 2
    }
}


class SwinIRPredictor(BasePredictor):

    def __init__(self,
                 output_path='output_dir',
                 weight_path=None,
                 seed=None,
                 window_size=8):
        self.output_path = output_path
        task = 'Denoising'
        self.task = task
        self.window_size = window_size

        if weight_path is None:
            if task in model_cfgs.keys():
                weight_path = get_path_from_url(model_cfgs[task]['model_urls'])
                checkpoint = paddle.load(weight_path)
            else:
                raise ValueError('Predictor need a task to define!')
        else:
            if weight_path.startswith("http"):  # os.path.islink dosen't work!
                weight_path = get_path_from_url(weight_path)
                checkpoint = paddle.load(weight_path)
            else:
                checkpoint = paddle.load(weight_path)

        self.generator = SwinIR(upscale=model_cfgs[task]['upscale'],
                                img_size=model_cfgs[task]['img_size'],
                                window_size=model_cfgs[task]['window_size'],
                                depths=model_cfgs[task]['depths'],
                                embed_dim=model_cfgs[task]['embed_dim'],
                                num_heads=model_cfgs[task]['num_heads'],
                                mlp_ratio=model_cfgs[task]['mlp_ratio'])

        checkpoint = checkpoint['generator']
        self.generator.set_state_dict(checkpoint)
        self.generator.eval()

        if seed is not None:
            paddle.seed(seed)
            random.seed(seed)
            np.random.seed(seed)

    def get_images(self, images_path):
        if os.path.isdir(images_path):
            return natsorted(
                glob(os.path.join(images_path, '*.jpeg')) +
                glob(os.path.join(images_path, '*.jpg')) +
                glob(os.path.join(images_path, '*.JPG')) +
                glob(os.path.join(images_path, '*.png')) +
                glob(os.path.join(images_path, '*.PNG')))
        else:
            return [images_path]

    def imread_uint(self, path, n_channels=3):
        #  input: path
        # output: HxWx3(RGB or GGG), or HxWx1 (G)
        if n_channels == 1:
            img = cv2.imread(path, 0)  # cv2.IMREAD_GRAYSCALE
            img = np.expand_dims(img, axis=2)  # HxWx1
        elif n_channels == 3:
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # BGR or G
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # GGG
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # RGB

        return img

    def uint2single(self, img):

        return np.float32(img / 255.)

    # convert single (HxWxC) to 3-dimensional paddle tensor
    def single2tensor3(self, img):
        return paddle.Tensor(np.ascontiguousarray(
            img, dtype=np.float32)).transpose([2, 0, 1])

    def run(self, images_path=None):
        os.makedirs(self.output_path, exist_ok=True)
        task_path = os.path.join(self.output_path, self.task)
        os.makedirs(task_path, exist_ok=True)
        image_files = self.get_images(images_path)
        for image_file in tqdm(image_files):
            img_L = self.imread_uint(image_file, 3)

            image_name = os.path.basename(image_file)
            img = cv2.cvtColor(img_L, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(task_path, image_name), img)

            tmps = image_name.split('.')
            assert len(
                tmps) == 2, f'Invalid image name: {image_name}, too much "."'
            restoration_save_path = os.path.join(
                task_path, f'{tmps[0]}_restoration.{tmps[1]}')

            img_L = self.uint2single(img_L)

            # HWC to CHW, numpy to tensor
            img_L = self.single2tensor3(img_L)
            img_L = img_L.unsqueeze(0)
            with paddle.no_grad():
                # pad input image to be a multiple of window_size
                _, _, h_old, w_old = img_L.shape
                h_pad = (h_old // self.window_size +
                         1) * self.window_size - h_old
                w_pad = (w_old // self.window_size +
                         1) * self.window_size - w_old
                img_L = paddle.concat([img_L, paddle.flip(img_L, [2])],
                                      2)[:, :, :h_old + h_pad, :]
                img_L = paddle.concat([img_L, paddle.flip(img_L, [3])],
                                      3)[:, :, :, :w_old + w_pad]
                output = self.generator(img_L)
                output = output[..., :h_old, :w_old]

            restored = paddle.clip(output, 0, 1)

            restored = restored.numpy()
            restored = restored.transpose(0, 2, 3, 1)
            restored = restored[0]
            restored = restored * 255
            restored = restored.astype(np.uint8)

            cv2.imwrite(restoration_save_path,
                        cv2.cvtColor(restored, cv2.COLOR_RGB2BGR))

        print('Done, output path is:', task_path)
