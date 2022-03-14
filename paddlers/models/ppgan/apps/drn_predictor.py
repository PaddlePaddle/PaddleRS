#  Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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
import numpy as np
from PIL import Image

import paddle 
from ppgan.models.generators import DRNGenerator
from ppgan.utils.download import get_path_from_url
from ppgan.utils.logger import get_logger

from .base_predictor import BasePredictor

REALSR_WEIGHT_URL = 'https://paddlegan.bj.bcebos.com/models/DRNSx4.pdparams'

class DRNPredictor(BasePredictor):
    def __init__(self, output='output', weight_path=None):
        self.input = input
        self.output = os.path.join(output, 'DRN') #定义超分的结果保存的路径，为output路径+模型名所在文件夹
        self.model = DRNGenerator((2, 4)) # 实例化模型
        if weight_path is None:
            weight_path = get_path_from_url(REALSR_WEIGHT_URL)
        state_dict = paddle.load(weight_path) #加载权重
        state_dict = state_dict['generator'] 
        self.model.load_dict(state_dict)
        self.model.eval()
    # 标准化
    def norm(self, img):
        img = np.array(img).transpose([2, 0, 1]).astype('float32') / 1.0
        return img.astype('float32')
    # 去标准化
    def denorm(self, img):
        img = img.transpose((1, 2, 0))
        return (img * 1).clip(0, 255).astype('uint8')

    # 对图片输入进行预测，输入可以是图像路径，也可以是cv2读取的矩阵，或者PIL读取的图像文件
    def run_image(self, img):
        if isinstance(img, str):
            ori_img = Image.open(img).convert('RGB')
        elif isinstance(img, np.ndarray):
            ori_img = Image.fromarray(img).convert('RGB')
        elif isinstance(img, Image.Image):
            ori_img = img

        img = self.norm(ori_img) #图像标准化
        x = paddle.to_tensor(img[np.newaxis, ...]) #转成tensor
        with paddle.no_grad():
            out = self.model(x)[2] # 执行预测，DRN模型会输出三个tensor，第一个是原始低分辨率影像，第二个是放大两倍，第三个才是我们所需要的最后的结果
            

        pred_img = self.denorm(out.numpy()[0]) #tensor转成numpy的array并去标准化
        pred_img = Image.fromarray(pred_img) # array转图像
        return pred_img

    #输入图像文件路径
    def run(self, input):
        # 如果输出路径不存在则新建一个
        if not os.path.exists(self.output):
            os.makedirs(self.output)

        pred_img = self.run_image(input) #对输入的图片进行预测
        out_path = None
        if self.output:
            try:
                base_name = os.path.splitext(os.path.basename(input))[0]
            except:
                base_name = 'result'
            out_path = os.path.join(self.output, base_name + '.png') #保存路径
            pred_img.save(out_path) #保存输出图片
            logger = get_logger()
            logger.info('Image saved to {}'.format(out_path))

        return pred_img, out_path
