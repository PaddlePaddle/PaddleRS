# copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np
import os.path as osp
import cv2
import copy
import random
import imghdr
from PIL import Image

try:
    from collections.abc import Sequence
except Exception:
    from collections import Sequence

# from paddlers.transforms.operators import Transform


class Transform(object):
    """
    Parent class of all data augmentation operations
    """

    def __init__(self):
        pass

    def apply_im(self, image):
        pass

    def apply_mask(self, mask):
        pass

    def apply_bbox(self, bbox):
        pass

    def apply_segm(self, segms):
        pass

    def apply(self, sample):
        sample['image'] = self.apply_im(sample['image'])
        if 'mask' in sample:
            sample['mask'] = self.apply_mask(sample['mask'])
        if 'gt_bbox' in sample:
            sample['gt_bbox'] = self.apply_bbox(sample['gt_bbox'])

        return sample

    def __call__(self, sample):
        if isinstance(sample, Sequence):
            sample = [self.apply(s) for s in sample]
        else:
            sample = self.apply(sample)

        return sample


class ImgDecode(Transform):
    """
    Decode image(s) in input.
    Args:
        to_rgb (bool, optional): If True, convert input images from BGR format to RGB format. Defaults to True.
    """

    def __init__(self, to_rgb=True):
        super(ImgDecode, self).__init__()
        self.to_rgb = to_rgb

    def read_img(self, img_path, input_channel=3):
        img_format = imghdr.what(img_path)
        name, ext = osp.splitext(img_path)
        if img_format == 'tiff' or ext == '.img':
            try:
                import gdal
            except:
                try:
                    from osgeo import gdal
                except:
                    raise Exception(
                        "Failed to import gdal! You can try use conda to install gdal"
                    )
                    six.reraise(*sys.exc_info())

            dataset = gdal.Open(img_path)
            if dataset == None:
                raise Exception('Can not open', img_path)
            im_data = dataset.ReadAsArray()
            return im_data.transpose((1, 2, 0))
        elif img_format in ['jpeg', 'bmp', 'png', 'jpg']:
            if input_channel == 3:
                return cv2.imread(img_path, cv2.IMREAD_ANYDEPTH |
                                  cv2.IMREAD_ANYCOLOR | cv2.IMREAD_COLOR)
            else:
                return cv2.imread(img_path, cv2.IMREAD_ANYDEPTH |
                                  cv2.IMREAD_ANYCOLOR)
        elif ext == '.npy':
            return np.load(img_path)
        else:
            raise Exception('Image format {} is not supported!'.format(ext))

    def apply_im(self, im_path):
        if isinstance(im_path, str):
            try:
                image = self.read_img(im_path)
            except:
                raise ValueError('Cannot read the image file {}!'.format(
                    im_path))
        else:
            image = im_path

        if self.to_rgb:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image

    def apply_mask(self, mask):
        try:
            mask = np.asarray(Image.open(mask))
        except:
            raise ValueError("Cannot read the mask file {}!".format(mask))
        if len(mask.shape) != 2:
            raise Exception(
                "Mask should be a 1-channel image, but recevied is a {}-channel image.".
                format(mask.shape[2]))
        return mask

    def apply(self, sample):
        """
        Args:
            sample (dict): Input sample, containing 'image' at least.
        Returns:
            dict: Decoded sample.
        """
        sample['image'] = self.apply_im(sample['image'])
        if 'mask' in sample:
            sample['mask'] = self.apply_mask(sample['mask'])
            im_height, im_width, _ = sample['image'].shape
            se_height, se_width = sample['mask'].shape
            if im_height != se_height or im_width != se_width:
                raise Exception(
                    "The height or width of the im is not same as the mask")

        sample['im_shape'] = np.array(
            sample['image'].shape[:2], dtype=np.float32)
        sample['scale_factor'] = np.array([1., 1.], dtype=np.float32)
        return sample