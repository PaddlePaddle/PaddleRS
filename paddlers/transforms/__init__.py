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
from .rotated_operators import *
from .batch_operators import BatchRandomResize, BatchRandomResizeByShort, _BatchPad, BatchNormalizeImage, BatchPadRGT, BatchCompose
from paddlers import transforms as T


def decode_image(im_path,
                 to_rgb=True,
                 to_uint8=True,
                 decode_bgr=True,
                 decode_sar=True,
                 read_geo_info=False,
                 use_stretch=False,
                 read_raw=False):
    """
    Decode an image.
    
    Args:
        im_path (str): Path of the image to decode.
        to_rgb (bool, optional): (Deprecated) If True, convert input image(s) from 
            BGR format to RGB format. Defaults to True.
        to_uint8 (bool, optional): If True, quantize and convert decoded image(s) to 
            uint8 type. Defaults to True.
        decode_bgr (bool, optional): If True, automatically interpret a non-geo 
            image (e.g. jpeg images) as a BGR image. Defaults to True.
        decode_sar (bool, optional): If True, automatically interpret a single-channel 
            geo image (e.g. geotiff images) as a SAR image, set this argument to 
            True. Defaults to True.
        read_geo_info (bool, optional): If True, read geographical information from 
            the image. Deafults to False.
        use_stretch (bool, optional): Whether to apply 2% linear stretch. Valid only if 
            `to_uint8` is True. Defaults to False.
        read_raw (bool, optional): If True, equivalent to setting `to_rgb` to `True` and
            `to_uint8` to False. Setting `read_raw` takes precedence over setting `to_rgb` 
            and `to_uint8`. Defaults to False.
    
    Returns:
        np.ndarray|tuple: If `read_geo_info` is False, return the decoded image. 
            Otherwise, return a tuple that contains the decoded image and a dictionary
            of geographical information (e.g. geographical transform and geographical 
            projection).
    """

    # Do a presence check. osp.exists() assumes `im_path` is a path-like object.
    if not osp.exists(im_path):
        raise ValueError(f"{im_path} does not exist!")
    if read_raw:
        to_rgb = True
        to_uint8 = False
    decoder = T.DecodeImg(
        to_rgb=to_rgb,
        to_uint8=to_uint8,
        decode_bgr=decode_bgr,
        decode_sar=decode_sar,
        read_geo_info=read_geo_info,
        use_stretch=use_stretch)
    # Deepcopy to avoid inplace modification
    sample = {'image': copy.deepcopy(im_path)}
    sample = decoder(sample)
    if read_geo_info:
        return sample['image'], sample['geo_info_dict']
    else:
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
