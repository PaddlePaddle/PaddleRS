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

import os
import copy
import random
from numbers import Number
from functools import partial
from operator import methodcaller
from collections.abc import Sequence

import numpy as np
import cv2
import imghdr
from PIL import Image
from joblib import load

import paddlers
import paddlers.transforms.indices as indices
from .functions import (
    normalize, horizontal_flip, permute, vertical_flip, center_crop, is_poly,
    horizontal_flip_poly, horizontal_flip_rle, vertical_flip_poly,
    vertical_flip_rle, crop_poly, crop_rle, expand_poly, expand_rle,
    resize_poly, resize_rle, dehaze, select_bands, to_intensity, to_uint8,
    img_flip, img_simple_rotate, decode_seg_mask, calc_hr_shape,
    match_by_regression, match_histograms)

__all__ = [
    "Compose",
    "DecodeImg",
    "Resize",
    "RandomResize",
    "ResizeByShort",
    "RandomResizeByShort",
    "ResizeByLong",
    "RandomHorizontalFlip",
    "RandomVerticalFlip",
    "Normalize",
    "CenterCrop",
    "RandomCrop",
    "RandomScaleAspect",
    "RandomExpand",
    "Pad",
    "MixupImage",
    "RandomDistort",
    "RandomBlur",
    "RandomSwap",
    "Dehaze",
    "ReduceDim",
    "SelectBand",
    "RandomFlipOrRotate",
    "ReloadMask",
    "AppendIndex",
    "MatchRadiance",
    "ArrangeRestorer",
    "ArrangeSegmenter",
    "ArrangeChangeDetector",
    "ArrangeClassifier",
    "ArrangeDetector",
]

interp_dict = {
    'NEAREST': cv2.INTER_NEAREST,
    'LINEAR': cv2.INTER_LINEAR,
    'CUBIC': cv2.INTER_CUBIC,
    'AREA': cv2.INTER_AREA,
    'LANCZOS4': cv2.INTER_LANCZOS4
}


class Compose(object):
    """
    Apply a series of data augmentation strategies to the input.
    All input images should be in Height-Width-Channel ([H, W, C]) format.

    Args:
        transforms (list[paddlers.transforms.Transform]): List of data preprocess or
            augmentation operators.

    Raises:
        TypeError: Invalid type of transforms.
        ValueError: Invalid length of transforms.
    """

    def __init__(self, transforms):
        super(Compose, self).__init__()
        if not isinstance(transforms, list):
            raise TypeError(
                "Type of transforms is invalid. Must be a list, but received is {}."
                .format(type(transforms)))
        if len(transforms) < 1:
            raise ValueError(
                "Length of transforms must not be less than 1, but received is {}."
                .format(len(transforms)))
        transforms = copy.deepcopy(transforms)
        self.arrange = self._pick_arrange(transforms)
        self.transforms = transforms

    def __call__(self, sample):
        """
        This is equivalent to sequentially calling compose_obj.apply_transforms() 
            and compose_obj.arrange_outputs().
        """

        sample = self.apply_transforms(sample)
        sample = self.arrange_outputs(sample)
        return sample

    def apply_transforms(self, sample):
        for op in self.transforms:
            # Skip batch transforms amd mixup
            if isinstance(op, (paddlers.transforms.BatchRandomResize,
                               paddlers.transforms.BatchRandomResizeByShort,
                               MixupImage)):
                continue
            sample = op(sample)
        return sample

    def arrange_outputs(self, sample):
        if self.arrange is not None:
            sample = self.arrange(sample)
        return sample

    def _pick_arrange(self, transforms):
        arrange = None
        for idx, op in enumerate(transforms):
            if isinstance(op, Arrange):
                if idx != len(transforms) - 1:
                    raise ValueError(
                        "Arrange operator must be placed at the end of the list."
                    )
                arrange = transforms.pop(idx)
        return arrange


class Transform(object):
    """
    Parent class of all data augmentation operators.
    """

    def __init__(self):
        pass

    def apply_im(self, image):
        return image

    def apply_mask(self, mask):
        return mask

    def apply_bbox(self, bbox):
        return bbox

    def apply_segm(self, segms):
        return segms

    def apply(self, sample):
        if 'image' in sample:
            sample['image'] = self.apply_im(sample['image'])
        else:  # image_tx
            sample['image'] = self.apply_im(sample['image_t1'])
            sample['image2'] = self.apply_im(sample['image_t2'])
        if 'mask' in sample:
            sample['mask'] = self.apply_mask(sample['mask'])
        if 'gt_bbox' in sample:
            sample['gt_bbox'] = self.apply_bbox(sample['gt_bbox'])
        if 'aux_masks' in sample:
            sample['aux_masks'] = list(
                map(self.apply_mask, sample['aux_masks']))
        if 'target' in sample:
            sample['target'] = self.apply_im(sample['target'])

        return sample

    def __call__(self, sample):
        if isinstance(sample, Sequence):
            sample = [self.apply(s) for s in sample]
        else:
            sample = self.apply(sample)

        return sample


class DecodeImg(Transform):
    """
    Decode image(s) in input.
    
    Args:
        to_rgb (bool, optional): If True, convert input image(s) from BGR format to 
            RGB format. Defaults to True.
        to_uint8 (bool, optional): If True, quantize and convert decoded image(s) to 
            uint8 type. Defaults to True.
        decode_bgr (bool, optional): If True, automatically interpret a non-geo image 
            (e.g., jpeg images) as a BGR image. Defaults to True.
        decode_sar (bool, optional): If True, automatically interpret a two-channel 
            geo image (e.g. geotiff images) as a SAR image, set this argument to 
            True. Defaults to True.
        read_geo_info (bool, optional): If True, read geographical information from 
            the image. Deafults to False.
    """

    def __init__(self,
                 to_rgb=True,
                 to_uint8=True,
                 decode_bgr=True,
                 decode_sar=True,
                 read_geo_info=False):
        super(DecodeImg, self).__init__()
        self.to_rgb = to_rgb
        self.to_uint8 = to_uint8
        self.decode_bgr = decode_bgr
        self.decode_sar = decode_sar
        self.read_geo_info = False

    def read_img(self, img_path):
        img_format = imghdr.what(img_path)
        name, ext = os.path.splitext(img_path)
        geo_trans, geo_proj = None, None

        if img_format == 'tiff' or ext == '.img':
            try:
                import gdal
            except:
                try:
                    from osgeo import gdal
                except ImportError:
                    raise ImportError(
                        "Failed to import gdal! Please install GDAL library according to the document."
                    )

            dataset = gdal.Open(img_path)
            if dataset == None:
                raise IOError('Cannot open', img_path)
            im_data = dataset.ReadAsArray()
            if im_data.ndim == 2 and self.decode_sar:
                im_data = to_intensity(im_data)
                im_data = im_data[:, :, np.newaxis]
            else:
                if im_data.ndim == 3:
                    im_data = im_data.transpose((1, 2, 0))
            if self.read_geo_info:
                geo_trans = dataset.GetGeoTransform()
                geo_proj = dataset.GetGeoProjection()
        elif img_format in ['jpeg', 'bmp', 'png', 'jpg']:
            if self.decode_bgr:
                im_data = cv2.imread(img_path, cv2.IMREAD_ANYDEPTH |
                                     cv2.IMREAD_ANYCOLOR | cv2.IMREAD_COLOR)
            else:
                im_data = cv2.imread(img_path, cv2.IMREAD_ANYDEPTH |
                                     cv2.IMREAD_ANYCOLOR)
        elif ext == '.npy':
            im_data = np.load(img_path)
        else:
            raise TypeError("Image format {} is not supported!".format(ext))

        if self.read_geo_info:
            return im_data, geo_trans, geo_proj
        else:
            return im_data

    def apply_im(self, im_path):
        if isinstance(im_path, str):
            try:
                data = self.read_img(im_path)
            except:
                raise ValueError("Cannot read the image file {}!".format(
                    im_path))
            if self.read_geo_info:
                image, geo_trans, geo_proj = data
                geo_info_dict = {'geo_trans': geo_trans, 'geo_proj': geo_proj}
            else:
                image = data
        else:
            image = im_path

        if self.to_rgb and image.shape[-1] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.to_uint8:
            image = to_uint8(image)

        if self.read_geo_info:
            return image, geo_info_dict
        else:
            return image

    def apply_mask(self, mask):
        try:
            mask = np.asarray(Image.open(mask))
        except:
            raise ValueError("Cannot read the mask file {}!".format(mask))
        if len(mask.shape) != 2:
            raise ValueError(
                "Mask should be a 1-channel image, but recevied is a {}-channel image.".
                format(mask.shape[2]))
        return mask

    def apply(self, sample):
        """
        Args:
            sample (dict): Input sample.

        Returns:
            dict: Sample with decoded images.
        """

        if 'image' in sample:
            if self.read_geo_info:
                image, geo_info_dict = self.apply_im(sample['image'])
                sample['image'] = image
                sample['geo_info_dict'] = geo_info_dict
            else:
                sample['image'] = self.apply_im(sample['image'])

        if 'image2' in sample:
            if self.read_geo_info:
                image2, geo_info_dict2 = self.apply_im(sample['image2'])
                sample['image2'] = image2
                sample['geo_info_dict2'] = geo_info_dict2
            else:
                sample['image2'] = self.apply_im(sample['image2'])

        if 'image_t1' in sample and not 'image' in sample:
            if not ('image_t2' in sample and 'image2' not in sample):
                raise ValueError
            if self.read_geo_info:
                image, geo_info_dict = self.apply_im(sample['image_t1'])
                sample['image'] = image
                sample['geo_info_dict'] = geo_info_dict
            else:
                sample['image'] = self.apply_im(sample['image_t1'])
            if self.read_geo_info:
                image2, geo_info_dict2 = self.apply_im(sample['image_t2'])
                sample['image2'] = image2
                sample['geo_info_dict2'] = geo_info_dict2
            else:
                sample['image2'] = self.apply_im(sample['image_t2'])

        if 'mask' in sample:
            sample['mask_ori'] = copy.deepcopy(sample['mask'])
            sample['mask'] = self.apply_mask(sample['mask'])
            im_height, im_width, _ = sample['image'].shape
            se_height, se_width = sample['mask'].shape
            if im_height != se_height or im_width != se_width:
                raise ValueError(
                    "The height or width of the image is not same as the mask.")

        if 'aux_masks' in sample:
            sample['aux_masks_ori'] = copy.deepcopy(sample['aux_masks'])
            sample['aux_masks'] = list(
                map(self.apply_mask, sample['aux_masks']))
            # TODO: check the shape of auxiliary masks

        if 'target' in sample:
            if self.read_geo_info:
                target, geo_info_dict = self.apply_im(sample['target'])
                sample['target'] = target
                sample['geo_info_dict_tar'] = geo_info_dict
            else:
                sample['target'] = self.apply_im(sample['target'])

        sample['im_shape'] = np.array(
            sample['image'].shape[:2], dtype=np.float32)
        sample['scale_factor'] = np.array([1., 1.], dtype=np.float32)

        return sample


class Resize(Transform):
    """
    Resize input.

    - If `target_size` is an int, resize the image(s) to (`target_size`, `target_size`).
    - If `target_size` is a list or tuple, resize the image(s) to `target_size`.
    Attention: If `interp` is 'RANDOM', the interpolation method will be chosen randomly.

    Args:
        target_size (int | list[int] | tuple[int]): Target size. If it is an integer, the
            target height and width will be both set to `target_size`. Otherwise, 
            `target_size` represents [target height, target width].
        interp (str, optional): Interpolation method for resizing image(s). One of 
            {'NEAREST', 'LINEAR', 'CUBIC', 'AREA', 'LANCZOS4', 'RANDOM'}. 
            Defaults to 'LINEAR'.
        keep_ratio (bool, optional): If True, the scaling factor of width and height will 
            be set to same value, and height/width of the resized image will be not 
            greater than the target width/height. Defaults to False.

    Raises:
        TypeError: Invalid type of target_size.
        ValueError: Invalid interpolation method.
    """

    def __init__(self, target_size, interp='LINEAR', keep_ratio=False):
        super(Resize, self).__init__()
        if not (interp == "RANDOM" or interp in interp_dict):
            raise ValueError("`interp` should be one of {}.".format(
                interp_dict.keys()))
        if isinstance(target_size, int):
            target_size = (target_size, target_size)
        else:
            if not (isinstance(target_size,
                               (list, tuple)) and len(target_size) == 2):
                raise TypeError(
                    "`target_size` should be an int or a list of length 2, but received {}.".
                    format(target_size))
        # (height, width)
        self.target_size = target_size
        self.interp = interp
        self.keep_ratio = keep_ratio

    def apply_im(self, image, interp, target_size):
        flag = image.shape[2] == 1
        image = cv2.resize(image, target_size, interpolation=interp)
        if flag:
            image = image[:, :, np.newaxis]
        return image

    def apply_mask(self, mask, target_size):
        mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)
        return mask

    def apply_bbox(self, bbox, scale, target_size):
        im_scale_x, im_scale_y = scale
        bbox[:, 0::2] *= im_scale_x
        bbox[:, 1::2] *= im_scale_y
        bbox[:, 0::2] = np.clip(bbox[:, 0::2], 0, target_size[0])
        bbox[:, 1::2] = np.clip(bbox[:, 1::2], 0, target_size[1])
        return bbox

    def apply_segm(self, segms, im_size, scale):
        im_h, im_w = im_size
        im_scale_x, im_scale_y = scale
        resized_segms = []
        for segm in segms:
            if is_poly(segm):
                # Polygon format
                resized_segms.append([
                    resize_poly(poly, im_scale_x, im_scale_y) for poly in segm
                ])
            else:
                # RLE format
                resized_segms.append(
                    resize_rle(segm, im_h, im_w, im_scale_x, im_scale_y))

        return resized_segms

    def apply(self, sample):
        if self.interp == "RANDOM":
            interp = random.choice(list(interp_dict.values()))
        else:
            interp = interp_dict[self.interp]
        im_h, im_w = sample['image'].shape[:2]

        im_scale_y = self.target_size[0] / im_h
        im_scale_x = self.target_size[1] / im_w
        target_size = (self.target_size[1], self.target_size[0])
        if self.keep_ratio:
            scale = min(im_scale_y, im_scale_x)
            target_w = int(round(im_w * scale))
            target_h = int(round(im_h * scale))
            target_size = (target_w, target_h)
            im_scale_y = target_h / im_h
            im_scale_x = target_w / im_w

        sample['image'] = self.apply_im(sample['image'], interp, target_size)
        if 'image2' in sample:
            sample['image2'] = self.apply_im(sample['image2'], interp,
                                             target_size)

        if 'mask' in sample:
            sample['mask'] = self.apply_mask(sample['mask'], target_size)
        if 'aux_masks' in sample:
            sample['aux_masks'] = list(
                map(partial(
                    self.apply_mask, target_size=target_size),
                    sample['aux_masks']))
        if 'gt_bbox' in sample and len(sample['gt_bbox']) > 0:
            sample['gt_bbox'] = self.apply_bbox(
                sample['gt_bbox'], [im_scale_x, im_scale_y], target_size)
        if 'gt_poly' in sample and len(sample['gt_poly']) > 0:
            sample['gt_poly'] = self.apply_segm(
                sample['gt_poly'], [im_h, im_w], [im_scale_x, im_scale_y])
        if 'target' in sample:
            if 'sr_factor' in sample:
                # For SR tasks
                sample['target'] = self.apply_im(
                    sample['target'], interp,
                    calc_hr_shape(target_size, sample['sr_factor']))
            else:
                # For non-SR tasks
                sample['target'] = self.apply_im(sample['target'], interp,
                                                 target_size)

        sample['im_shape'] = np.asarray(
            sample['image'].shape[:2], dtype=np.float32)
        if 'scale_factor' in sample:
            scale_factor = sample['scale_factor']
            sample['scale_factor'] = np.asarray(
                [scale_factor[0] * im_scale_y, scale_factor[1] * im_scale_x],
                dtype=np.float32)
        return sample


class RandomResize(Transform):
    """
    Resize input to random sizes.

    Attention: If `interp` is 'RANDOM', the interpolation method will be chosen randomly.

    Args:
        target_sizes (list[int] | list[list|tuple] | tuple[list|tuple]):
            Multiple target sizes, each of which should be int, list, or tuple.
        interp (str, optional): Interpolation method for resizing image(s). One of 
            {'NEAREST', 'LINEAR', 'CUBIC', 'AREA', 'LANCZOS4', 'RANDOM'}. 
            Defaults to 'LINEAR'.

    Raises:
        TypeError: Invalid type of `target_size`.
        ValueError: Invalid interpolation method.
    """

    def __init__(self, target_sizes, interp='LINEAR'):
        super(RandomResize, self).__init__()
        if not (interp == "RANDOM" or interp in interp_dict):
            raise ValueError("`interp` should be one of {}.".format(
                interp_dict.keys()))
        self.interp = interp
        assert isinstance(target_sizes, list), \
            "`target_size` must be a list."
        for i, item in enumerate(target_sizes):
            if isinstance(item, int):
                target_sizes[i] = (item, item)
        self.target_size = target_sizes

    def apply(self, sample):
        height, width = random.choice(self.target_size)
        resizer = Resize((height, width), interp=self.interp)
        sample = resizer(sample)

        return sample


class ResizeByShort(Transform):
    """
    Resize input while keeping the aspect ratio.

    Attention: If `interp` is 'RANDOM', the interpolation method will be chosen randomly.

    Args:
        short_size (int): Target size of the shorter side of the image(s).
        max_size (int, optional): Upper bound of longer side of the image(s). If
            `max_size` is -1, no upper bound will be applied. Defaults to -1.
        interp (str, optional): Interpolation method for resizing image(s). One of 
            {'NEAREST', 'LINEAR', 'CUBIC', 'AREA', 'LANCZOS4', 'RANDOM'}. 
            Defaults to 'LINEAR'.

    Raises:
        ValueError: Invalid interpolation method.
    """

    def __init__(self, short_size=256, max_size=-1, interp='LINEAR'):
        if not (interp == "RANDOM" or interp in interp_dict):
            raise ValueError("`interp` should be one of {}".format(
                interp_dict.keys()))
        super(ResizeByShort, self).__init__()
        self.short_size = short_size
        self.max_size = max_size
        self.interp = interp

    def apply(self, sample):
        im_h, im_w = sample['image'].shape[:2]
        im_short_size = min(im_h, im_w)
        im_long_size = max(im_h, im_w)
        scale = float(self.short_size) / float(im_short_size)
        if 0 < self.max_size < np.round(scale * im_long_size):
            scale = float(self.max_size) / float(im_long_size)
        target_w = int(round(im_w * scale))
        target_h = int(round(im_h * scale))
        sample = Resize(
            target_size=(target_h, target_w), interp=self.interp)(sample)

        return sample


class RandomResizeByShort(Transform):
    """
    Resize input to random sizes while keeping the aspect ratio.

    Attention: If `interp` is 'RANDOM', the interpolation method will be chosen randomly.

    Args:
        short_sizes (list[int]): Target size of the shorter side of the image(s).
        max_size (int, optional): Upper bound of longer side of the image(s). 
            If `max_size` is -1, no upper bound will be applied. Defaults to -1.
        interp (str, optional): Interpolation method for resizing image(s). One of 
            {'NEAREST', 'LINEAR', 'CUBIC', 'AREA', 'LANCZOS4', 'RANDOM'}. 
            Defaults to 'LINEAR'.

    Raises:
        TypeError: Invalid type of `target_size`.
        ValueError: Invalid interpolation method.

    See Also:
        ResizeByShort: Resize image(s) in input while keeping the aspect ratio.
    """

    def __init__(self, short_sizes, max_size=-1, interp='LINEAR'):
        super(RandomResizeByShort, self).__init__()
        if not (interp == "RANDOM" or interp in interp_dict):
            raise ValueError("`interp` should be one of {}".format(
                interp_dict.keys()))
        self.interp = interp
        assert isinstance(short_sizes, list), \
            "`short_sizes` must be a list."

        self.short_sizes = short_sizes
        self.max_size = max_size

    def apply(self, sample):
        short_size = random.choice(self.short_sizes)
        resizer = ResizeByShort(
            short_size=short_size, max_size=self.max_size, interp=self.interp)
        sample = resizer(sample)
        return sample


class ResizeByLong(Transform):
    def __init__(self, long_size=256, interp='LINEAR'):
        super(ResizeByLong, self).__init__()
        self.long_size = long_size
        self.interp = interp

    def apply(self, sample):
        im_h, im_w = sample['image'].shape[:2]
        im_long_size = max(im_h, im_w)
        scale = float(self.long_size) / float(im_long_size)
        target_h = int(round(im_h * scale))
        target_w = int(round(im_w * scale))
        sample = Resize(
            target_size=(target_h, target_w), interp=self.interp)(sample)

        return sample


class RandomFlipOrRotate(Transform):
    """
    Flip or Rotate an image in different directions with a certain probability.

    Args:
        probs (list[float]): Probabilities of performing flipping and rotation. 
            Default: [0.35,0.25].
        probsf (list[float]): Probabilities of 5 flipping modes (horizontal, 
            vertical, both horizontal and vertical, diagonal, anti-diagonal). 
            Default: [0.3, 0.3, 0.2, 0.1, 0.1].
        probsr (list[float]): Probabilities of 3 rotation modes (90°, 180°, 270° 
            clockwise). Default: [0.25, 0.5, 0.25].

    Examples:

        from paddlers import transforms as T

        # Define operators for data augmentation
        train_transforms = T.Compose([
            T.DecodeImg(),
            T.RandomFlipOrRotate(
                probs  = [0.3, 0.2]             # p=0.3 to flip the image，p=0.2 to rotate the image，p=0.5 to keep the image unchanged.
                probsf = [0.3, 0.25, 0, 0, 0]   # p=0.3 and p=0.25 to perform horizontal and vertical flipping; probility of no-flipping is 0.45.
                probsr = [0, 0.65, 0]),         # p=0.65 to rotate the image by 180°; probility of no-rotation is 0.35.
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    """

    def __init__(self,
                 probs=[0.35, 0.25],
                 probsf=[0.3, 0.3, 0.2, 0.1, 0.1],
                 probsr=[0.25, 0.5, 0.25]):
        super(RandomFlipOrRotate, self).__init__()
        # Change various probabilities into probability intervals, to judge in which mode to flip or rotate
        self.probs = [probs[0], probs[0] + probs[1]]
        self.probsf = self.get_probs_range(probsf)
        self.probsr = self.get_probs_range(probsr)

    def apply_im(self, image, mode_id, flip_mode=True):
        if flip_mode:
            image = img_flip(image, mode_id)
        else:
            image = img_simple_rotate(image, mode_id)
        return image

    def apply_mask(self, mask, mode_id, flip_mode=True):
        if flip_mode:
            mask = img_flip(mask, mode_id)
        else:
            mask = img_simple_rotate(mask, mode_id)
        return mask

    def apply_bbox(self, bbox, mode_id, flip_mode=True):
        raise TypeError(
            "Currently, RandomFlipOrRotate is not available for object detection tasks."
        )

    def apply_segm(self, bbox, mode_id, flip_mode=True):
        raise TypeError(
            "Currently, RandomFlipOrRotate is not available for object detection tasks."
        )

    def get_probs_range(self, probs):
        """
        Change list of probabilities into cumulative probability intervals.

        Args:
            probs (list[float]): Probabilities of different modes, shape: [n].

        Returns:
            list[list]: Probability intervals, shape: [n, 2].
        """

        ps = []
        last_prob = 0
        for prob in probs:
            p_s = last_prob
            cur_prob = prob / sum(probs)
            last_prob += cur_prob
            p_e = last_prob
            ps.append([p_s, p_e])
        return ps

    def judge_probs_range(self, p, probs):
        """
        Judge whether the value of `p` falls within the given probability interval.

        Args:
            p (float): Value between 0 and 1.
            probs (list[list]): Probability intervals, shape: [n, 2].

        Returns:
            int: Interval where the input probability falls into.
        """

        for id, id_range in enumerate(probs):
            if p > id_range[0] and p < id_range[1]:
                return id
        return -1

    def apply(self, sample):
        p_m = random.random()
        if p_m < self.probs[0]:
            mode_p = random.random()
            mode_id = self.judge_probs_range(mode_p, self.probsf)
            sample['image'] = self.apply_im(sample['image'], mode_id, True)
            if 'image2' in sample:
                sample['image2'] = self.apply_im(sample['image2'], mode_id,
                                                 True)
            if 'mask' in sample:
                sample['mask'] = self.apply_mask(sample['mask'], mode_id, True)
            if 'aux_masks' in sample:
                sample['aux_masks'] = [
                    self.apply_mask(aux_mask, mode_id, True)
                    for aux_mask in sample['aux_masks']
                ]
            if 'gt_bbox' in sample and len(sample['gt_bbox']) > 0:
                sample['gt_bbox'] = self.apply_bbox(sample['gt_bbox'], mode_id,
                                                    True)
            if 'gt_poly' in sample and len(sample['gt_poly']) > 0:
                sample['gt_poly'] = self.apply_segm(sample['gt_poly'], mode_id,
                                                    True)
            if 'target' in sample:
                sample['target'] = self.apply_im(sample['target'], mode_id,
                                                 True)
        elif p_m < self.probs[1]:
            mode_p = random.random()
            mode_id = self.judge_probs_range(mode_p, self.probsr)
            sample['image'] = self.apply_im(sample['image'], mode_id, False)
            if 'image2' in sample:
                sample['image2'] = self.apply_im(sample['image2'], mode_id,
                                                 False)
            if 'mask' in sample:
                sample['mask'] = self.apply_mask(sample['mask'], mode_id, False)
            if 'aux_masks' in sample:
                sample['aux_masks'] = [
                    self.apply_mask(aux_mask, mode_id, False)
                    for aux_mask in sample['aux_masks']
                ]
            if 'gt_bbox' in sample and len(sample['gt_bbox']) > 0:
                sample['gt_bbox'] = self.apply_bbox(sample['gt_bbox'], mode_id,
                                                    False)
            if 'gt_poly' in sample and len(sample['gt_poly']) > 0:
                sample['gt_poly'] = self.apply_segm(sample['gt_poly'], mode_id,
                                                    False)
            if 'target' in sample:
                sample['target'] = self.apply_im(sample['target'], mode_id,
                                                 False)

        return sample


class RandomHorizontalFlip(Transform):
    """
    Randomly flip the input horizontally.

    Args:
        prob (float, optional): Probability of flipping the input. Defaults to .5.
    """

    def __init__(self, prob=0.5):
        super(RandomHorizontalFlip, self).__init__()
        self.prob = prob

    def apply_im(self, image):
        image = horizontal_flip(image)
        return image

    def apply_mask(self, mask):
        mask = horizontal_flip(mask)
        return mask

    def apply_bbox(self, bbox, width):
        oldx1 = bbox[:, 0].copy()
        oldx2 = bbox[:, 2].copy()
        bbox[:, 0] = width - oldx2
        bbox[:, 2] = width - oldx1
        return bbox

    def apply_segm(self, segms, height, width):
        flipped_segms = []
        for segm in segms:
            if is_poly(segm):
                # Polygon format
                flipped_segms.append(
                    [horizontal_flip_poly(poly, width) for poly in segm])
            else:
                # RLE format
                flipped_segms.append(horizontal_flip_rle(segm, height, width))
        return flipped_segms

    def apply(self, sample):
        if random.random() < self.prob:
            im_h, im_w = sample['image'].shape[:2]
            sample['image'] = self.apply_im(sample['image'])
            if 'image2' in sample:
                sample['image2'] = self.apply_im(sample['image2'])
            if 'mask' in sample:
                sample['mask'] = self.apply_mask(sample['mask'])
            if 'aux_masks' in sample:
                sample['aux_masks'] = list(
                    map(self.apply_mask, sample['aux_masks']))
            if 'gt_bbox' in sample and len(sample['gt_bbox']) > 0:
                sample['gt_bbox'] = self.apply_bbox(sample['gt_bbox'], im_w)
            if 'gt_poly' in sample and len(sample['gt_poly']) > 0:
                sample['gt_poly'] = self.apply_segm(sample['gt_poly'], im_h,
                                                    im_w)
            if 'target' in sample:
                sample['target'] = self.apply_im(sample['target'])
        return sample


class RandomVerticalFlip(Transform):
    """
    Randomly flip the input vertically.

    Args:
        prob (float, optional): Probability of flipping the input. Defaults to .5.
    """

    def __init__(self, prob=0.5):
        super(RandomVerticalFlip, self).__init__()
        self.prob = prob

    def apply_im(self, image):
        image = vertical_flip(image)
        return image

    def apply_mask(self, mask):
        mask = vertical_flip(mask)
        return mask

    def apply_bbox(self, bbox, height):
        oldy1 = bbox[:, 1].copy()
        oldy2 = bbox[:, 3].copy()
        bbox[:, 0] = height - oldy2
        bbox[:, 2] = height - oldy1
        return bbox

    def apply_segm(self, segms, height, width):
        flipped_segms = []
        for segm in segms:
            if is_poly(segm):
                # Polygon format
                flipped_segms.append(
                    [vertical_flip_poly(poly, height) for poly in segm])
            else:
                # RLE format
                flipped_segms.append(vertical_flip_rle(segm, height, width))
        return flipped_segms

    def apply(self, sample):
        if random.random() < self.prob:
            im_h, im_w = sample['image'].shape[:2]
            sample['image'] = self.apply_im(sample['image'])
            if 'image2' in sample:
                sample['image2'] = self.apply_im(sample['image2'])
            if 'mask' in sample:
                sample['mask'] = self.apply_mask(sample['mask'])
            if 'aux_masks' in sample:
                sample['aux_masks'] = list(
                    map(self.apply_mask, sample['aux_masks']))
            if 'gt_bbox' in sample and len(sample['gt_bbox']) > 0:
                sample['gt_bbox'] = self.apply_bbox(sample['gt_bbox'], im_h)
            if 'gt_poly' in sample and len(sample['gt_poly']) > 0:
                sample['gt_poly'] = self.apply_segm(sample['gt_poly'], im_h,
                                                    im_w)
            if 'target' in sample:
                sample['target'] = self.apply_im(sample['target'])
        return sample


class Normalize(Transform):
    """
    Apply normalization to the input image(s). The normalization steps are:
    1. im = (im - min_value) * 1 / (max_value - min_value)
    2. im = im - mean
    3. im = im / std

    Args:
        mean (list[float] | tuple[float], optional): Mean of input image(s). 
            Defaults to [0.485, 0.456, 0.406].
        std (list[float] | tuple[float], optional): Standard deviation of input 
            image(s). Defaults to [0.229, 0.224, 0.225].
        min_val (list[float] | tuple[float], optional): Minimum value of input 
            image(s). If None, use 0 for all channels. Defaults to None.
        max_val (list[float] | tuple[float], optional): Maximum value of input 
            image(s). If None, use 255. for all channels. Defaults to None.
        apply_to_tar (bool, optional): Whether to apply transformation to the target
            image. Defaults to True.
    """

    def __init__(self,
                 mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225],
                 min_val=None,
                 max_val=None,
                 apply_to_tar=True):
        super(Normalize, self).__init__()
        channel = len(mean)
        if min_val is None:
            min_val = [0] * channel
        if max_val is None:
            max_val = [255.] * channel

        from functools import reduce
        if reduce(lambda x, y: x * y, std) == 0:
            raise ValueError(
                "`std` should not contain 0, but received is {}.".format(std))
        if reduce(lambda x, y: x * y,
                  [a - b for a, b in zip(max_val, min_val)]) == 0:
            raise ValueError(
                "(`max_val` - `min_val`) should not contain 0, but received is {}.".
                format((np.asarray(max_val) - np.asarray(min_val)).tolist()))

        self.mean = mean
        self.std = std
        self.min_val = min_val
        self.max_val = max_val
        self.apply_to_tar = apply_to_tar

    def apply_im(self, image):
        image = image.astype(np.float32)
        mean = np.asarray(
            self.mean, dtype=np.float32)[np.newaxis, np.newaxis, :]
        std = np.asarray(self.std, dtype=np.float32)[np.newaxis, np.newaxis, :]
        image = normalize(image, mean, std, self.min_val, self.max_val)
        return image

    def apply(self, sample):
        sample['image'] = self.apply_im(sample['image'])
        if 'image2' in sample:
            sample['image2'] = self.apply_im(sample['image2'])
        if 'target' in sample and self.apply_to_tar:
            sample['target'] = self.apply_im(sample['target'])

        return sample


class CenterCrop(Transform):
    """
    Crop the input image(s) at the center.
    1. Locate the center of the image.
    2. Crop the image.

    Args:
        crop_size (int, optional): Target size of the cropped image(s). 
            Defaults to 224.
    """

    def __init__(self, crop_size=224):
        super(CenterCrop, self).__init__()
        self.crop_size = crop_size

    def apply_im(self, image):
        image = center_crop(image, self.crop_size)

        return image

    def apply_mask(self, mask):
        mask = center_crop(mask, self.crop_size)
        return mask

    def apply(self, sample):
        sample['image'] = self.apply_im(sample['image'])
        if 'image2' in sample:
            sample['image2'] = self.apply_im(sample['image2'])
        if 'mask' in sample:
            sample['mask'] = self.apply_mask(sample['mask'])
        if 'aux_masks' in sample:
            sample['aux_masks'] = list(
                map(self.apply_mask, sample['aux_masks']))
        if 'target' in sample:
            sample['target'] = self.apply_im(sample['target'])
        return sample


class RandomCrop(Transform):
    """
    Randomly crop the input.
    1. Compute the height and width of cropped area according to `aspect_ratio` and 
        `scaling`.
    2. Locate the upper left corner of cropped area randomly.
    3. Crop the image(s).
    4. Resize the cropped area to `crop_size` x `crop_size`.

    Args:
        crop_size (int | list[int] | tuple[int]): Target size of the cropped area. If 
            None, the cropped area will not be resized. Defaults to None.
        aspect_ratio (list[float], optional): Aspect ratio of cropped region in 
            [min, max] format. Defaults to [.5, 2.].
        thresholds (list[float], optional): Iou thresholds to decide a valid bbox 
            crop. Defaults to [.0, .1, .3, .5, .7, .9].
        scaling (list[float], optional): Ratio between the cropped region and the 
            original image in [min, max] format. Defaults to [.3, 1.].
        num_attempts (int, optional): Max number of tries before giving up. 
            Defaults to 50.
        allow_no_crop (bool, optional): Whether returning without doing crop is 
            allowed. Defaults to True.
        cover_all_box (bool, optional): Whether to ensure all bboxes be covered in 
            the final crop. Defaults to False.
    """

    def __init__(self,
                 crop_size=None,
                 aspect_ratio=[.5, 2.],
                 thresholds=[.0, .1, .3, .5, .7, .9],
                 scaling=[.3, 1.],
                 num_attempts=50,
                 allow_no_crop=True,
                 cover_all_box=False):
        super(RandomCrop, self).__init__()
        self.crop_size = crop_size
        self.aspect_ratio = aspect_ratio
        self.thresholds = thresholds
        self.scaling = scaling
        self.num_attempts = num_attempts
        self.allow_no_crop = allow_no_crop
        self.cover_all_box = cover_all_box

    def _generate_crop_info(self, sample):
        im_h, im_w = sample['image'].shape[:2]
        if 'gt_bbox' in sample and len(sample['gt_bbox']) > 0:
            thresholds = self.thresholds
            if self.allow_no_crop:
                thresholds.append('no_crop')
            np.random.shuffle(thresholds)
            for thresh in thresholds:
                if thresh == 'no_crop':
                    return None
                for i in range(self.num_attempts):
                    crop_box = self._get_crop_box(im_h, im_w)
                    if crop_box is None:
                        continue
                    iou = self._iou_matrix(
                        sample['gt_bbox'],
                        np.array(
                            [crop_box], dtype=np.float32))
                    if iou.max() < thresh:
                        continue
                    if self.cover_all_box and iou.min() < thresh:
                        continue
                    cropped_box, valid_ids = self._crop_box_with_center_constraint(
                        sample['gt_bbox'], np.array(
                            crop_box, dtype=np.float32))
                    if valid_ids.size > 0:
                        return crop_box, cropped_box, valid_ids
        else:
            for i in range(self.num_attempts):
                crop_box = self._get_crop_box(im_h, im_w)
                if crop_box is None:
                    continue
                return crop_box, None, None
        return None

    def _get_crop_box(self, im_h, im_w):
        scale = np.random.uniform(*self.scaling)
        if self.aspect_ratio is not None:
            min_ar, max_ar = self.aspect_ratio
            aspect_ratio = np.random.uniform(
                max(min_ar, scale**2), min(max_ar, scale**-2))
            h_scale = scale / np.sqrt(aspect_ratio)
            w_scale = scale * np.sqrt(aspect_ratio)
        else:
            h_scale = np.random.uniform(*self.scaling)
            w_scale = np.random.uniform(*self.scaling)
        crop_h = im_h * h_scale
        crop_w = im_w * w_scale
        if self.aspect_ratio is None:
            if crop_h / crop_w < 0.5 or crop_h / crop_w > 2.0:
                return None
        crop_h = int(crop_h)
        crop_w = int(crop_w)
        crop_y = np.random.randint(0, im_h - crop_h)
        crop_x = np.random.randint(0, im_w - crop_w)
        return [crop_x, crop_y, crop_x + crop_w, crop_y + crop_h]

    def _iou_matrix(self, a, b):
        tl_i = np.maximum(a[:, np.newaxis, :2], b[:, :2])
        br_i = np.minimum(a[:, np.newaxis, 2:], b[:, 2:])

        area_i = np.prod(br_i - tl_i, axis=2) * (tl_i < br_i).all(axis=2)
        area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
        area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
        area_o = (area_a[:, np.newaxis] + area_b - area_i)
        return area_i / (area_o + 1e-10)

    def _crop_box_with_center_constraint(self, box, crop):
        cropped_box = box.copy()

        cropped_box[:, :2] = np.maximum(box[:, :2], crop[:2])
        cropped_box[:, 2:] = np.minimum(box[:, 2:], crop[2:])
        cropped_box[:, :2] -= crop[:2]
        cropped_box[:, 2:] -= crop[:2]

        centers = (box[:, :2] + box[:, 2:]) / 2
        valid = np.logical_and(crop[:2] <= centers,
                               centers < crop[2:]).all(axis=1)
        valid = np.logical_and(
            valid, (cropped_box[:, :2] < cropped_box[:, 2:]).all(axis=1))

        return cropped_box, np.where(valid)[0]

    def _crop_segm(self, segms, valid_ids, crop, height, width):
        crop_segms = []
        for id in valid_ids:
            segm = segms[id]
            if is_poly(segm):
                # Polygon format
                crop_segms.append(crop_poly(segm, crop))
            else:
                # RLE format
                crop_segms.append(crop_rle(segm, crop, height, width))

        return crop_segms

    def apply_im(self, image, crop):
        x1, y1, x2, y2 = crop
        return image[y1:y2, x1:x2, :]

    def apply_mask(self, mask, crop):
        x1, y1, x2, y2 = crop
        return mask[y1:y2, x1:x2, ...]

    def apply(self, sample):
        crop_info = self._generate_crop_info(sample)
        if crop_info is not None:
            crop_box, cropped_box, valid_ids = crop_info
            im_h, im_w = sample['image'].shape[:2]
            sample['image'] = self.apply_im(sample['image'], crop_box)
            if 'image2' in sample:
                sample['image2'] = self.apply_im(sample['image2'], crop_box)
            if 'gt_poly' in sample and len(sample['gt_poly']) > 0:
                crop_polys = self._crop_segm(
                    sample['gt_poly'],
                    valid_ids,
                    np.array(
                        crop_box, dtype=np.int64),
                    im_h,
                    im_w)
                if [] in crop_polys:
                    delete_id = list()
                    valid_polys = list()
                    for idx, poly in enumerate(crop_polys):
                        if not crop_poly:
                            delete_id.append(idx)
                        else:
                            valid_polys.append(poly)
                    valid_ids = np.delete(valid_ids, delete_id)
                    if not valid_polys:
                        return sample
                    sample['gt_poly'] = valid_polys
                else:
                    sample['gt_poly'] = crop_polys

            if 'gt_bbox' in sample and len(sample['gt_bbox']) > 0:
                sample['gt_bbox'] = np.take(cropped_box, valid_ids, axis=0)
                sample['gt_class'] = np.take(
                    sample['gt_class'], valid_ids, axis=0)
                if 'gt_score' in sample:
                    sample['gt_score'] = np.take(
                        sample['gt_score'], valid_ids, axis=0)
                if 'is_crowd' in sample:
                    sample['is_crowd'] = np.take(
                        sample['is_crowd'], valid_ids, axis=0)

            if 'mask' in sample:
                sample['mask'] = self.apply_mask(sample['mask'], crop_box)

            if 'aux_masks' in sample:
                sample['aux_masks'] = list(
                    map(partial(
                        self.apply_mask, crop=crop_box),
                        sample['aux_masks']))

            if 'target' in sample:
                if 'sr_factor' in sample:
                    sample['target'] = self.apply_im(
                        sample['target'],
                        calc_hr_shape(crop_box, sample['sr_factor']))
                else:
                    sample['target'] = self.apply_im(sample['image'], crop_box)

        if self.crop_size is not None:
            sample = Resize(self.crop_size)(sample)

        return sample


class RandomScaleAspect(Transform):
    """
    Crop input image(s) and resize back to original sizes.

    Args: 
        min_scale (float): Minimum ratio between the cropped region and the original
            image. If 0, image(s) will not be cropped. Defaults to .5.
        aspect_ratio (float): Aspect ratio of cropped region. Defaults to .33.
    """

    def __init__(self, min_scale=0.5, aspect_ratio=0.33):
        super(RandomScaleAspect, self).__init__()
        self.min_scale = min_scale
        self.aspect_ratio = aspect_ratio

    def apply(self, sample):
        if self.min_scale != 0 and self.aspect_ratio != 0:
            img_height, img_width = sample['image'].shape[:2]
            sample = RandomCrop(
                crop_size=(img_height, img_width),
                aspect_ratio=[self.aspect_ratio, 1. / self.aspect_ratio],
                scaling=[self.min_scale, 1.],
                num_attempts=10,
                allow_no_crop=False)(sample)
        return sample


class RandomExpand(Transform):
    """
    Randomly expand the input by padding according to random offsets.

    Args:
        upper_ratio (float, optional): Maximum ratio to which the original image 
            is expanded. Defaults to 4..
        prob (float, optional): Probability of apply expanding. Defaults to .5.
        im_padding_value (list[float] | tuple[float], optional): RGB filling value 
            for the image. Defaults to (127.5, 127.5, 127.5).
        label_padding_value (int, optional): Filling value for the mask. 
            Defaults to 255.

    See Also:
        paddlers.transforms.Pad
    """

    def __init__(self,
                 upper_ratio=4.,
                 prob=.5,
                 im_padding_value=127.5,
                 label_padding_value=255):
        super(RandomExpand, self).__init__()
        assert upper_ratio > 1.01, "`upper_ratio` must be larger than 1.01."
        self.upper_ratio = upper_ratio
        self.prob = prob
        assert isinstance(im_padding_value, (Number, Sequence)), \
            "Value to fill must be either float or sequence."
        self.im_padding_value = im_padding_value
        self.label_padding_value = label_padding_value

    def apply(self, sample):
        if random.random() < self.prob:
            im_h, im_w = sample['image'].shape[:2]
            ratio = np.random.uniform(1., self.upper_ratio)
            h = int(im_h * ratio)
            w = int(im_w * ratio)
            if h > im_h and w > im_w:
                y = np.random.randint(0, h - im_h)
                x = np.random.randint(0, w - im_w)
                target_size = (h, w)
                offsets = (x, y)
                sample = Pad(
                    target_size=target_size,
                    pad_mode=-1,
                    offsets=offsets,
                    im_padding_value=self.im_padding_value,
                    label_padding_value=self.label_padding_value)(sample)
        return sample


class Pad(Transform):
    def __init__(self,
                 target_size=None,
                 pad_mode=0,
                 offsets=None,
                 im_padding_value=127.5,
                 label_padding_value=255,
                 size_divisor=32):
        """
        Pad image to a specified size or multiple of `size_divisor`.

        Args:
            target_size (list[int] | tuple[int], optional): Image target size, if None, pad to 
                multiple of size_divisor. Defaults to None.
            pad_mode (int, optional): Pad mode. Currently only four modes are supported:
                [-1, 0, 1, 2]. if -1, use specified offsets. If 0, only pad to right and bottom
                If 1, pad according to center. If 2, only pad left and top. Defaults to 0.
            offsets (list[int]|None, optional): Padding offsets. Defaults to None.
            im_padding_value (list[float] | tuple[float]): RGB value of padded area. 
                Defaults to (127.5, 127.5, 127.5).
            label_padding_value (int, optional): Filling value for the mask. 
                Defaults to 255.
            size_divisor (int): Image width and height after padding will be a multiple of 
                `size_divisor`.
        """
        super(Pad, self).__init__()
        if isinstance(target_size, (list, tuple)):
            if len(target_size) != 2:
                raise ValueError(
                    "`target_size` should contain 2 elements, but it is {}.".
                    format(target_size))
        if isinstance(target_size, int):
            target_size = [target_size] * 2

        assert pad_mode in [
            -1, 0, 1, 2
        ], "Currently only four modes are supported: [-1, 0, 1, 2]."
        if pad_mode == -1:
            assert offsets, "if `pad_mode` is -1, `offsets` should not be None."

        self.target_size = target_size
        self.size_divisor = size_divisor
        self.pad_mode = pad_mode
        self.offsets = offsets
        self.im_padding_value = im_padding_value
        self.label_padding_value = label_padding_value

    def apply_im(self, image, offsets, target_size):
        x, y = offsets
        h, w = target_size
        im_h, im_w, channel = image.shape[:3]
        canvas = np.ones((h, w, channel), dtype=np.float32)
        canvas *= np.array(self.im_padding_value, dtype=np.float32)
        canvas[y:y + im_h, x:x + im_w, :] = image.astype(np.float32)
        return canvas

    def apply_mask(self, mask, offsets, target_size):
        x, y = offsets
        im_h, im_w = mask.shape[:2]
        h, w = target_size
        canvas = np.ones((h, w), dtype=np.float32)
        canvas *= np.array(self.label_padding_value, dtype=np.float32)
        canvas[y:y + im_h, x:x + im_w] = mask.astype(np.float32)
        return canvas

    def apply_bbox(self, bbox, offsets):
        return bbox + np.array(offsets * 2, dtype=np.float32)

    def apply_segm(self, segms, offsets, im_size, size):
        x, y = offsets
        height, width = im_size
        h, w = size
        expanded_segms = []
        for segm in segms:
            if is_poly(segm):
                # Polygon format
                expanded_segms.append(
                    [expand_poly(poly, x, y) for poly in segm])
            else:
                # RLE format
                expanded_segms.append(
                    expand_rle(segm, x, y, height, width, h, w))
        return expanded_segms

    def _get_offsets(self, im_h, im_w, h, w):
        if self.pad_mode == -1:
            offsets = self.offsets
        elif self.pad_mode == 0:
            offsets = [0, 0]
        elif self.pad_mode == 1:
            offsets = [(w - im_w) // 2, (h - im_h) // 2]
        else:
            offsets = [w - im_w, h - im_h]
        return offsets

    def apply(self, sample):
        im_h, im_w = sample['image'].shape[:2]
        if self.target_size:
            h, w = self.target_size
            assert (
                    im_h <= h and im_w <= w
            ), 'target size ({}, {}) cannot be less than image size ({}, {})'\
                .format(h, w, im_h, im_w)
        else:
            h = (np.ceil(im_h / self.size_divisor) *
                 self.size_divisor).astype(int)
            w = (np.ceil(im_w / self.size_divisor) *
                 self.size_divisor).astype(int)

        if h == im_h and w == im_w:
            return sample

        offsets = self._get_offsets(im_h, im_w, h, w)

        sample['image'] = self.apply_im(sample['image'], offsets, (h, w))
        if 'image2' in sample:
            sample['image2'] = self.apply_im(sample['image2'], offsets, (h, w))
        if 'mask' in sample:
            sample['mask'] = self.apply_mask(sample['mask'], offsets, (h, w))
        if 'aux_masks' in sample:
            sample['aux_masks'] = list(
                map(partial(
                    self.apply_mask, offsets=offsets, target_size=(h, w)),
                    sample['aux_masks']))
        if 'gt_bbox' in sample and len(sample['gt_bbox']) > 0:
            sample['gt_bbox'] = self.apply_bbox(sample['gt_bbox'], offsets)
        if 'gt_poly' in sample and len(sample['gt_poly']) > 0:
            sample['gt_poly'] = self.apply_segm(
                sample['gt_poly'], offsets, im_size=[im_h, im_w], size=[h, w])
        if 'target' in sample:
            if 'sr_factor' in sample:
                hr_shape = calc_hr_shape((h, w), sample['sr_factor'])
                hr_offsets = self._get_offsets(*sample['target'].shape[:2],
                                               *hr_shape)
                sample['target'] = self.apply_im(sample['target'], hr_offsets,
                                                 hr_shape)
            else:
                sample['target'] = self.apply_im(sample['target'], offsets,
                                                 (h, w))
        return sample


class MixupImage(Transform):
    def __init__(self, alpha=1.5, beta=1.5, mixup_epoch=-1):
        """
        Mixup two images and their gt_bbbox/gt_score.

        Args:
            alpha (float, optional): Alpha parameter of beta distribution. 
                Defaults to 1.5.
            beta (float, optional): Beta parameter of beta distribution. 
                Defaults to 1.5.
        """
        super(MixupImage, self).__init__()
        if alpha <= 0.0:
            raise ValueError("`alpha` should be positive in MixupImage.")
        if beta <= 0.0:
            raise ValueError("`beta` should be positive in MixupImage.")
        self.alpha = alpha
        self.beta = beta
        self.mixup_epoch = mixup_epoch

    def apply_im(self, image1, image2, factor):
        h = max(image1.shape[0], image2.shape[0])
        w = max(image1.shape[1], image2.shape[1])
        img = np.zeros((h, w, image1.shape[2]), 'float32')
        img[:image1.shape[0], :image1.shape[1], :] = \
            image1.astype('float32') * factor
        img[:image2.shape[0], :image2.shape[1], :] += \
            image2.astype('float32') * (1.0 - factor)
        return img.astype('uint8')

    def __call__(self, sample):
        if not isinstance(sample, Sequence):
            return sample

        assert len(sample) == 2, 'mixup need two samples'

        factor = np.random.beta(self.alpha, self.beta)
        factor = max(0.0, min(1.0, factor))
        if factor >= 1.0:
            return sample[0]
        if factor <= 0.0:
            return sample[1]
        image = self.apply_im(sample[0]['image'], sample[1]['image'], factor)
        result = copy.deepcopy(sample[0])
        result['image'] = image
        # Apply bbox and score
        if 'gt_bbox' in sample[0]:
            gt_bbox1 = sample[0]['gt_bbox']
            gt_bbox2 = sample[1]['gt_bbox']
            gt_bbox = np.concatenate((gt_bbox1, gt_bbox2), axis=0)
            result['gt_bbox'] = gt_bbox
        if 'gt_poly' in sample[0]:
            gt_poly1 = sample[0]['gt_poly']
            gt_poly2 = sample[1]['gt_poly']
            gt_poly = gt_poly1 + gt_poly2
            result['gt_poly'] = gt_poly
        if 'gt_class' in sample[0]:
            gt_class1 = sample[0]['gt_class']
            gt_class2 = sample[1]['gt_class']
            gt_class = np.concatenate((gt_class1, gt_class2), axis=0)
            result['gt_class'] = gt_class

            gt_score1 = np.ones_like(sample[0]['gt_class'])
            gt_score2 = np.ones_like(sample[1]['gt_class'])
            gt_score = np.concatenate(
                (gt_score1 * factor, gt_score2 * (1. - factor)), axis=0)
            result['gt_score'] = gt_score
        if 'is_crowd' in sample[0]:
            is_crowd1 = sample[0]['is_crowd']
            is_crowd2 = sample[1]['is_crowd']
            is_crowd = np.concatenate((is_crowd1, is_crowd2), axis=0)
            result['is_crowd'] = is_crowd
        if 'difficult' in sample[0]:
            is_difficult1 = sample[0]['difficult']
            is_difficult2 = sample[1]['difficult']
            is_difficult = np.concatenate(
                (is_difficult1, is_difficult2), axis=0)
            result['difficult'] = is_difficult

        return result


class RandomDistort(Transform):
    """
    Random color distortion.

    Args:
        brightness_range (float, optional): Range of brightness distortion. 
            Defaults to .5.
        brightness_prob (float, optional): Probability of brightness distortion. 
            Defaults to .5.
        contrast_range (float, optional): Range of contrast distortion. 
            Defaults to .5.
        contrast_prob (float, optional): Probability of contrast distortion. 
            Defaults to .5.
        saturation_range (float, optional): Range of saturation distortion. 
            Defaults to .5.
        saturation_prob (float, optional): Probability of saturation distortion. 
            Defaults to .5.
        hue_range (float, optional): Range of hue distortion. Defaults to .5.
        hue_prob (float, optional): Probability of hue distortion. Defaults to .5.
        random_apply (bool, optional): Apply the transformation in random (yolo) or
            fixed (SSD) order. Defaults to True.
        count (int, optional): Number of distortions to apply. Defaults to 4.
        shuffle_channel (bool, optional): Whether to swap channels randomly. 
            Defaults to False.
    """

    def __init__(self,
                 brightness_range=0.5,
                 brightness_prob=0.5,
                 contrast_range=0.5,
                 contrast_prob=0.5,
                 saturation_range=0.5,
                 saturation_prob=0.5,
                 hue_range=18,
                 hue_prob=0.5,
                 random_apply=True,
                 count=4,
                 shuffle_channel=False):
        super(RandomDistort, self).__init__()
        self.brightness_range = [1 - brightness_range, 1 + brightness_range]
        self.brightness_prob = brightness_prob
        self.contrast_range = [1 - contrast_range, 1 + contrast_range]
        self.contrast_prob = contrast_prob
        self.saturation_range = [1 - saturation_range, 1 + saturation_range]
        self.saturation_prob = saturation_prob
        self.hue_range = [1 - hue_range, 1 + hue_range]
        self.hue_prob = hue_prob
        self.random_apply = random_apply
        self.count = count
        self.shuffle_channel = shuffle_channel

    def apply_hue(self, image):
        low, high = self.hue_range
        if np.random.uniform(0., 1.) < self.hue_prob:
            return image

        # It works, but the result differs from HSV version.
        delta = np.random.uniform(low, high)
        u = np.cos(delta * np.pi)
        w = np.sin(delta * np.pi)
        bt = np.array([[1.0, 0.0, 0.0], [0.0, u, -w], [0.0, w, u]])
        tyiq = np.array([[0.299, 0.587, 0.114], [0.596, -0.274, -0.321],
                         [0.211, -0.523, 0.311]])
        ityiq = np.array([[1.0, 0.956, 0.621], [1.0, -0.272, -0.647],
                          [1.0, -1.107, 1.705]])
        t = np.dot(np.dot(ityiq, bt), tyiq).T

        res_list = []
        channel = image.shape[2]
        for i in range(channel // 3):
            sub_img = image[:, :, 3 * i:3 * (i + 1)]
            sub_img = sub_img.astype(np.float32)
            sub_img = np.dot(image, t)
            res_list.append(sub_img)

        if channel % 3 != 0:
            i = channel % 3
            res_list.append(image[:, :, -i:])

        return np.concatenate(res_list, axis=2)

    def apply_saturation(self, image):
        low, high = self.saturation_range
        delta = np.random.uniform(low, high)
        if np.random.uniform(0., 1.) < self.saturation_prob:
            return image

        res_list = []
        channel = image.shape[2]
        for i in range(channel // 3):
            sub_img = image[:, :, 3 * i:3 * (i + 1)]
            sub_img = sub_img.astype(np.float32)
            # It works, but the result differs from HSV version.
            gray = sub_img * np.array(
                [[[0.299, 0.587, 0.114]]], dtype=np.float32)
            gray = gray.sum(axis=2, keepdims=True)
            gray *= (1.0 - delta)
            sub_img *= delta
            sub_img += gray
            res_list.append(sub_img)

        if channel % 3 != 0:
            i = channel % 3
            res_list.append(image[:, :, -i:])

        return np.concatenate(res_list, axis=2)

    def apply_contrast(self, image):
        low, high = self.contrast_range
        if np.random.uniform(0., 1.) < self.contrast_prob:
            return image
        delta = np.random.uniform(low, high)
        image = image.astype(np.float32)
        image *= delta
        return image

    def apply_brightness(self, image):
        low, high = self.brightness_range
        if np.random.uniform(0., 1.) < self.brightness_prob:
            return image
        delta = np.random.uniform(low, high)
        image = image.astype(np.float32)
        image += delta
        return image

    def apply(self, sample):
        if self.random_apply:
            functions = [
                self.apply_brightness, self.apply_contrast,
                self.apply_saturation, self.apply_hue
            ]
            distortions = np.random.permutation(functions)[:self.count]
            for func in distortions:
                sample['image'] = func(sample['image'])
                if 'image2' in sample:
                    sample['image2'] = func(sample['image2'])
            return sample

        sample['image'] = self.apply_brightness(sample['image'])
        if 'image2' in sample:
            sample['image2'] = self.apply_brightness(sample['image2'])
        mode = np.random.randint(0, 2)
        if mode:
            sample['image'] = self.apply_contrast(sample['image'])
            if 'image2' in sample:
                sample['image2'] = self.apply_contrast(sample['image2'])
        sample['image'] = self.apply_saturation(sample['image'])
        sample['image'] = self.apply_hue(sample['image'])
        if 'image2' in sample:
            sample['image2'] = self.apply_saturation(sample['image2'])
            sample['image2'] = self.apply_hue(sample['image2'])
        if not mode:
            sample['image'] = self.apply_contrast(sample['image'])
            if 'image2' in sample:
                sample['image2'] = self.apply_contrast(sample['image2'])

        if self.shuffle_channel:
            if np.random.randint(0, 2):
                sample['image'] = sample['image'][..., np.random.permutation(3)]
                if 'image2' in sample:
                    sample['image2'] = sample['image2'][
                        ..., np.random.permutation(3)]
        return sample


class RandomBlur(Transform):
    """
    Randomly blur input image(s).

    Args: 
        prob (float): Probability of blurring.
    """

    def __init__(self, prob=0.1):
        super(RandomBlur, self).__init__()
        self.prob = prob

    def apply_im(self, image, radius):
        image = cv2.GaussianBlur(image, (radius, radius), 0, 0)
        return image

    def apply(self, sample):
        if self.prob <= 0:
            n = 0
        elif self.prob >= 1:
            n = 1
        else:
            n = int(1.0 / self.prob)
        if n > 0:
            if np.random.randint(0, n) == 0:
                radius = np.random.randint(3, 10)
                if radius % 2 != 1:
                    radius = radius + 1
                if radius > 9:
                    radius = 9
                sample['image'] = self.apply_im(sample['image'], radius)
                if 'image2' in sample:
                    sample['image2'] = self.apply_im(sample['image2'], radius)
        return sample


class Dehaze(Transform):
    """
    Dehaze input image(s).

    Args: 
        gamma (bool, optional): Use gamma correction or not. Defaults to False.
    """

    def __init__(self, gamma=False):
        super(Dehaze, self).__init__()
        self.gamma = gamma

    def apply_im(self, image):
        image = dehaze(image, self.gamma)
        return image

    def apply(self, sample):
        sample['image'] = self.apply_im(sample['image'])
        if 'image2' in sample:
            sample['image2'] = self.apply_im(sample['image2'])
        return sample


class ReduceDim(Transform):
    """
    Use PCA to reduce the dimension of input image(s).

    Args: 
        joblib_path (str): Path of *.joblib file of PCA.
        apply_to_tar (bool, optional): Whether to apply transformation to the target
            image. Defaults to True.
    """

    def __init__(self, joblib_path, apply_to_tar=True):
        super(ReduceDim, self).__init__()
        ext = joblib_path.split(".")[-1]
        if ext != "joblib":
            raise ValueError("`joblib_path` must be *.joblib, not *.{}.".format(
                ext))
        self.pca = load(joblib_path)
        self.apply_to_tar = apply_to_tar

    def apply_im(self, image):
        H, W, C = image.shape
        n_im = np.reshape(image, (-1, C))
        im_pca = self.pca.transform(n_im)
        result = np.reshape(im_pca, (H, W, -1))
        return result

    def apply(self, sample):
        sample['image'] = self.apply_im(sample['image'])
        if 'image2' in sample:
            sample['image2'] = self.apply_im(sample['image2'])
        if 'target' in sample and self.apply_to_tar:
            sample['target'] = self.apply_im(sample['target'])
        return sample


class SelectBand(Transform):
    """
    Select a set of bands of input image(s).

    Args: 
        band_list (list, optional): Bands to select (band index starts from 1). 
            Defaults to [1, 2, 3].
        apply_to_tar (bool, optional): Whether to apply transformation to the target
            image. Defaults to True.
    """

    def __init__(self, band_list=[1, 2, 3], apply_to_tar=True):
        super(SelectBand, self).__init__()
        self.band_list = band_list
        self.apply_to_tar = apply_to_tar

    def apply_im(self, image):
        image = select_bands(image, self.band_list)
        return image

    def apply(self, sample):
        sample['image'] = self.apply_im(sample['image'])
        if 'image2' in sample:
            sample['image2'] = self.apply_im(sample['image2'])
        if 'target' in sample and self.apply_to_tar:
            sample['target'] = self.apply_im(sample['target'])
        return sample


class _PadBox(Transform):
    def __init__(self, num_max_boxes=50):
        """
        Pad zeros to bboxes if number of bboxes is less than `num_max_boxes`.

        Args:
            num_max_boxes (int, optional): Max number of bboxes. Defaults to 50.
        """

        self.num_max_boxes = num_max_boxes
        super(_PadBox, self).__init__()

    def apply(self, sample):
        gt_num = min(self.num_max_boxes, len(sample['gt_bbox']))
        num_max = self.num_max_boxes
        pad_bbox = np.zeros((num_max, 4), dtype=np.float32)
        if gt_num > 0:
            pad_bbox[:gt_num, :] = sample['gt_bbox'][:gt_num, :]
        sample['gt_bbox'] = pad_bbox
        if 'gt_class' in sample:
            pad_class = np.zeros((num_max, ), dtype=np.int32)
            if gt_num > 0:
                pad_class[:gt_num] = sample['gt_class'][:gt_num, 0]
            sample['gt_class'] = pad_class
        if 'gt_score' in sample:
            pad_score = np.zeros((num_max, ), dtype=np.float32)
            if gt_num > 0:
                pad_score[:gt_num] = sample['gt_score'][:gt_num, 0]
            sample['gt_score'] = pad_score
        # In training, for example in op ExpandImage,
        # bbox and gt_class are expanded, but difficult is not,
        # so judge by its length.
        if 'difficult' in sample:
            pad_diff = np.zeros((num_max, ), dtype=np.int32)
            if gt_num > 0:
                pad_diff[:gt_num] = sample['difficult'][:gt_num, 0]
            sample['difficult'] = pad_diff
        if 'is_crowd' in sample:
            pad_crowd = np.zeros((num_max, ), dtype=np.int32)
            if gt_num > 0:
                pad_crowd[:gt_num] = sample['is_crowd'][:gt_num, 0]
            sample['is_crowd'] = pad_crowd
        return sample


class _NormalizeBox(Transform):
    def __init__(self):
        super(_NormalizeBox, self).__init__()

    def apply(self, sample):
        height, width = sample['image'].shape[:2]
        for i in range(sample['gt_bbox'].shape[0]):
            sample['gt_bbox'][i][0] = sample['gt_bbox'][i][0] / width
            sample['gt_bbox'][i][1] = sample['gt_bbox'][i][1] / height
            sample['gt_bbox'][i][2] = sample['gt_bbox'][i][2] / width
            sample['gt_bbox'][i][3] = sample['gt_bbox'][i][3] / height

        return sample


class _BboxXYXY2XYWH(Transform):
    """
    Convert bbox XYXY format to XYWH format.
    """

    def __init__(self):
        super(_BboxXYXY2XYWH, self).__init__()

    def apply(self, sample):
        bbox = sample['gt_bbox']
        bbox[:, 2:4] = bbox[:, 2:4] - bbox[:, :2]
        bbox[:, :2] = bbox[:, :2] + bbox[:, 2:4] / 2.
        sample['gt_bbox'] = bbox
        return sample


class _Permute(Transform):
    def __init__(self):
        super(_Permute, self).__init__()

    def apply(self, sample):
        sample['image'] = permute(sample['image'], False)
        if 'image2' in sample:
            sample['image2'] = permute(sample['image2'], False)
        if 'target' in sample:
            sample['target'] = permute(sample['target'], False)
        return sample


class RandomSwap(Transform):
    """
    Randomly swap multi-temporal images.

    Args:
        prob (float, optional): Probability of swapping the input images. 
            Default: 0.2.
    """

    def __init__(self, prob=0.2):
        super(RandomSwap, self).__init__()
        self.prob = prob

    def apply(self, sample):
        if 'image2' not in sample:
            raise ValueError("'image2' is not found in the sample.")
        if random.random() < self.prob:
            sample['image'], sample['image2'] = sample['image2'], sample[
                'image']
        return sample


class ReloadMask(Transform):
    def apply(self, sample):
        sample['mask'] = decode_seg_mask(sample['mask_ori'])
        if 'aux_masks' in sample:
            sample['aux_masks'] = list(
                map(decode_seg_mask, sample['aux_masks_ori']))
        return sample


class AppendIndex(Transform):
    """
    Append remote sensing index to input image(s).

    Args:
        index_type (str): Type of remote sensinng index. See supported 
            index types in 
            https://github.com/PaddlePaddle/PaddleRS/tree/develop/paddlers/transforms/indices.py .
        band_indices (dict): Mapping of band names to band indices 
            (starting from 1). See band names in 
            https://github.com/PaddlePaddle/PaddleRS/tree/develop/paddlers/transforms/indices.py . 
    """

    def __init__(self, index_type, band_indices, **kwargs):
        super(AppendIndex, self).__init__()
        cls = getattr(indices, index_type)
        self._compute_index = cls(band_indices, **kwargs)

    def apply_im(self, image):
        index = self._compute_index(image)
        index = index[..., None].astype('float32')
        return np.concatenate([image, index], axis=-1)

    def apply(self, sample):
        sample['image'] = self.apply_im(sample['image'])
        if 'image2' in sample:
            sample['image2'] = self.apply_im(sample['image2'])
        return sample


class MatchRadiance(Transform):
    """
    Perform relative radiometric correction between bi-temporal images.

    Args:
        method (str, optional): Method used to match the radiance of the
            bi-temporal images. Choices are {'hist', 'lsr'}. 'hist' stands
            for histogram matching and 'lsr' stands for least-squares 
            regression. Default: 'hist'.
    """

    def __init__(self, method='hist'):
        super(MatchRadiance, self).__init__()

        if method == 'hist':
            self._match_func = match_histograms
        elif method == 'lsr':
            self._match_func = match_by_regression
        else:
            raise ValueError(
                "{} is not a supported radiometric correction method.".format(
                    method))

        self.method = method

    def apply(self, sample):
        if 'image2' not in sample:
            raise ValueError("'image2' is not found in the sample.")

        sample['image2'] = self._match_func(sample['image2'], sample['image'])
        return sample


class Arrange(Transform):
    def __init__(self, mode):
        super().__init__()
        if mode not in ['train', 'eval', 'test', 'quant']:
            raise ValueError(
                "`mode` should be defined as one of ['train', 'eval', 'test', 'quant']!"
            )
        self.mode = mode


class ArrangeSegmenter(Arrange):
    def apply(self, sample):
        if 'mask' in sample:
            mask = sample['mask']
            mask = mask.astype('int64')

        image = permute(sample['image'], False)
        if self.mode == 'train':
            return image, mask
        if self.mode == 'eval':
            return image, mask
        if self.mode == 'test':
            return image,


class ArrangeChangeDetector(Arrange):
    def apply(self, sample):
        if 'mask' in sample:
            mask = sample['mask']
            mask = mask.astype('int64')

        image_t1 = permute(sample['image'], False)
        image_t2 = permute(sample['image2'], False)
        if self.mode == 'train':
            masks = [mask]
            if 'aux_masks' in sample:
                masks.extend(
                    map(methodcaller('astype', 'int64'), sample['aux_masks']))
            return (
                image_t1,
                image_t2, ) + tuple(masks)
        if self.mode == 'eval':
            return image_t1, image_t2, mask
        if self.mode == 'test':
            return image_t1, image_t2,


class ArrangeClassifier(Arrange):
    def apply(self, sample):
        image = permute(sample['image'], False)
        if self.mode in ['train', 'eval']:
            return image, sample['label']
        else:
            return image


class ArrangeDetector(Arrange):
    def apply(self, sample):
        if self.mode == 'eval' and 'gt_poly' in sample:
            del sample['gt_poly']
        return sample


class ArrangeRestorer(Arrange):
    def apply(self, sample):
        if 'target' in sample:
            target = permute(sample['target'], False)
        image = permute(sample['image'], False)
        if self.mode == 'train':
            return image, target
        if self.mode == 'eval':
            return image, target
        if self.mode == 'test':
            return image,
