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

import copy
import os.path as osp

from .operators import *
from .batch_operators import BatchRandomResize, BatchRandomResizeByShort, _BatchPad
from paddlers import transforms as T


def decode_image(im_path,
                 to_rgb=True,
                 to_uint8=True,
                 decode_bgr=True,
                 decode_sar=True):
    """
    Decode an image.
    
    Args:
        im_path (str): Path of the image to decode.
        to_rgb (bool, optional): If True, convert input image(s) from BGR format to RGB format. Defaults to True.
        to_uint8 (bool, optional): If True, quantize and convert decoded image(s) to uint8 type. Defaults to True.
        decode_bgr (bool, optional): If True, automatically interpret a non-geo image (e.g. jpeg images) as a BGR image. 
            Defaults to True.
        decode_sar (bool, optional): If True, automatically interpret a two-channel geo image (e.g. geotiff images) as a 
            SAR image, set this argument to True. Defaults to True.

    Returns:
        np.ndarray: Decoded image.
    """

    # Do a presence check. osp.exists() assumes `im_path` is a path-like object.
    if not osp.exists(im_path):
        raise ValueError(f"{im_path} does not exist!")
    decoder = T.DecodeImg(
        to_rgb=to_rgb,
        to_uint8=to_uint8,
        decode_bgr=decode_bgr,
        decode_sar=decode_sar)
    # Deepcopy to avoid inplace modification
    sample = {'image': copy.deepcopy(im_path)}
    sample = decoder(sample)
    return sample['image']


def build_transforms(transforms_info):
    transforms = list()
    for op_info in transforms_info:
        op_name = list(op_info.keys())[0]
        op_attr = op_info[op_name]
        if not hasattr(T, op_name):
            raise ValueError(
                "There is no transform operator named '{}'.".format(op_name))
        transforms.append(getattr(T, op_name)(**op_attr))
    eval_transforms = T.Compose(transforms)
    return eval_transforms
