# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

try:
    from collections.abc import Sequence
except Exception:
    from collections import Sequence
from numbers import Integral

import numpy as np
import cv2

from .operators import Transform
from .rbox_utils import poly2rbox_le135_np, poly2rbox_oc_np, rbox2poly_np


class Poly2Array(Transform):
    """
    Convert gt_poly to np.array for rotated bboxes.
    """

    def __init__(self):
        super(Poly2Array, self).__init__()

    def apply_segm(self, poly):
        return np.array(poly, dtype=np.float32).reshape((-1, 8))


class Poly2RBox(Transform):
    """
    Convert Polygon to Rotated Box, using new OpenCV definition since 4.5.1.

    Args:
        filter_threshold (int|float, optional): Threshold to filter annotations.
            Default: 4.
        filter_mode (str, optional): Filtering mode: 'area', 'edge', or None.
            Default: None.
        rbox_type (str, optional): RBox type: 'le135' or 'oc'. Default: 'le135'.
    """

    def __init__(self, filter_threshold=4, filter_mode=None, rbox_type="le135"):
        super(Poly2RBox, self).__init__()
        self.filter_fn = lambda size: self.filter(size, filter_threshold, filter_mode)
        self.rbox_fn = poly2rbox_le135_np if rbox_type == "le135" else poly2rbox_oc_np

    def filter(self, size, threshold, mode):
        if mode == "area":
            if size[0] * size[1] < threshold:
                return True
        elif mode == "edge":
            if min(size) < threshold:
                return True
        return False

    def get_rbox(self, polys):
        valid_ids, rboxes, bboxes = [], [], []
        for i, poly in enumerate(polys):
            cx, cy, w, h, angle = self.rbox_fn(poly)
            if self.filter_fn((w, h)):
                continue
            rboxes.append(np.array([cx, cy, w, h, angle], dtype=np.float32))
            valid_ids.append(i)
            xmin, ymin = min(poly[0::2]), min(poly[1::2])
            xmax, ymax = max(poly[0::2]), max(poly[1::2])
            bboxes.append(np.array([xmin, ymin, xmax, ymax], dtype=np.float32))

        if len(valid_ids) == 0:
            rboxes = np.zeros((0, 5), dtype=np.float32)
            bboxes = np.zeros((0, 4), dtype=np.float32)
        else:
            rboxes = np.stack(rboxes)
            bboxes = np.stack(bboxes)

        return rboxes, bboxes, valid_ids

    def apply(self, sample):
        rboxes, bboxes, valid_ids = self.get_rbox(sample["gt_poly"])
        sample["gt_rbox"] = rboxes
        sample["gt_bbox"] = bboxes
        for k in ["gt_class", "gt_score", "gt_poly", "is_crowd", "difficult"]:
            if k in sample:
                sample[k] = sample[k][valid_ids]

        return sample


class RandomRFlip(Transform):
    """
    Randomly horizontally flip an image and its corresponding 
        bounding boxes and polygons.
    
    Args:
        prob (float, optional): Probability of flipping the image.
            Default: 0.5.
    """

    def __init__(self, prob=0.5):
        super(RandomRFlip, self).__init__()
        self.prob = prob
        if not (isinstance(self.prob, float)):
            raise TypeError("{}: input type is invalid.".format(self))

    def apply_im(self, image):
        """
        Flips the image horizontally.

        Args:
            image (numpy.ndarray): Input image.
        
        Returns:
            numpy.ndarray: Horizontally flipped image.
        """
        return image[:, ::-1, :]

    def apply_pts(self, pts, width):
        """
        Flips the bounding boxes and polygons horizontally.

        Args:
            pts (numpy.ndarray): Array of points representing bounding 
                boxes or polygons.
            width (int): Width of the original image.
        
        Returns:
            numpy.ndarray: Horizontally flipped bounding boxes or polygons.
        """
        oldx = pts[:, 0::2].copy()
        pts[:, 0::2] = width - oldx - 1
        return pts

    def apply(self, sample):
        if np.random.uniform(0, 1) < self.prob:
            im = sample["image"]
            _, width = im.shape[:2]
            im = self.apply_im(im)
            if "gt_bbox" in sample and len(sample["gt_bbox"]) > 0:
                sample["gt_bbox"] = self.apply_pts(sample["gt_bbox"], width)
            if "gt_poly" in sample and len(sample["gt_poly"]) > 0:
                sample["gt_poly"] = self.apply_pts(sample["gt_poly"], width)

            sample["flipped"] = True
            sample["image"] = im
        return sample


class RRotate(Transform):
    """
    Applies a rotation transformation to an image and its corresponding 
        bounding boxes and polygons.

    Args:
        scale (float, optional): Scale factor for the rotation. Default: 1.0.
        angle (float, optional): Angle of rotation in degrees. Default: 0.0.
        fill_value (float, optional): Value to fill pixels outside the image. 
            Default: 0.0.
        auto_bound (bool, optional): Whether to automatically adjust the output 
            image size to accommodate the entire rotated image. Default is True.
    """

    def __init__(self, scale=1.0, angle=0.0, fill_value=0.0, auto_bound=True):
        super(RRotate, self).__init__()
        self.scale = scale
        self.angle = angle
        self.fill_value = fill_value
        self.auto_bound = auto_bound

    def get_rotated_matrix(self, angle, scale, h, w):
        center = ((w - 1) * 0.5, (h - 1) * 0.5)
        matrix = cv2.getRotationMatrix2D(center, -angle, scale)
        # calculate the new size
        cos = np.abs(matrix[0, 0])
        sin = np.abs(matrix[0, 1])
        new_w = h * sin + w * cos
        new_h = h * cos + w * sin
        # calculate offset
        n_w = int(np.round(new_w))
        n_h = int(np.round(new_h))
        if self.auto_bound:
            ratio = min(w / n_w, h / n_h)
            matrix = cv2.getRotationMatrix2D(center, -angle, ratio)
        else:
            matrix[0, 2] += (new_w - w) * 0.5
            matrix[1, 2] += (new_h - h) * 0.5
            w = n_w
            h = n_h
        return matrix, h, w

    def get_rect_from_pts(self, pts, h, w):
        """get minimum rectangle of points"""
        assert pts.shape[-1] % 2 == 0, "the dim of input [pts] is not correct"
        min_x, min_y = np.min(pts[:, 0::2], axis=1), np.min(pts[:, 1::2],
                                                            axis=1)
        max_x, max_y = np.max(pts[:, 0::2], axis=1), np.max(pts[:, 1::2],
                                                            axis=1)
        min_x, min_y = np.clip(min_x, 0, w), np.clip(min_y, 0, h)
        max_x, max_y = np.clip(max_x, 0, w), np.clip(max_y, 0, h)
        boxes = np.stack([min_x, min_y, max_x, max_y], axis=-1)
        return boxes

    def apply_im(self, image, matrix, h, w):
        return cv2.warpAffine(
            image, matrix, (w, h), borderValue=self.fill_value)

    def apply_pts(self, pts, matrix):
        assert pts.shape[-1] % 2 == 0, "the dim of input [pts] is not correct"
        # n is number of samples and m is two times the number of points due to (x, y)
        _, m = pts.shape
        # transpose points
        pts_ = pts.reshape(-1, 2).T
        # pad 1 to convert the points to homogeneous coordinates
        padding = np.ones((1, pts_.shape[1]), pts.dtype)
        rotated_pts = np.matmul(matrix, np.concatenate((pts_, padding), axis=0))
        return rotated_pts[:2, :].T.reshape(-1, m)

    def apply(self, sample):
        image = sample["image"]
        h, w = image.shape[:2]
        matrix, h, w = self.get_rotated_matrix(self.angle, self.scale, h, w)
        sample["image"] = self.apply_im(image, matrix, h, w)
        polys = sample["gt_poly"]
        # TODO: segment or keypoint to be processed
        if len(polys) > 0:
            pts = self.apply_pts(polys, matrix)
            sample["gt_poly"] = pts
            sample["gt_bbox"] = self.get_rect_from_pts(pts, h, w)

        return sample


class RandomRRotate(Transform):
    """
    Apply a random rotation transformation to an image and its corresponding 
        bounding boxes and polygons.

    Args:
        scale (float|list, optional): Scale factor or range of scale factors 
            for the rotation. Default: 1.0.
        scale_mode (str, optional): Mode for selecting the scale factor. 
            Must be one of 'range', 'value', or None. Default: None.
        angle (float|list, optional): Angle or range of angles in degrees 
            for the rotation. Default: 0.0.
        angle_mode (str, optional): Mode for selecting the rotation angle. 
            Must be one of 'range', 'value', or None. Default is None.
        fill_value (float, optional): Value to fill pixels outside the image. 
            Default: 0.0.
        rotate_prob (float, optional): Probability of applying the rotation 
            transformation. Default: 1.0.
        auto_bound (bool, optional): Whether to automatically adjust the output image 
            size to accommodate the entire rotated image. Default is True.
    """

    def __init__(
            self,
            scale=1.0,
            scale_mode=None,
            angle=0.0,
            angle_mode=None,
            fill_value=0.0,
            rotate_prob=1.0,
            auto_bound=True, ):
        super(RandomRRotate, self).__init__()
        assert not angle_mode or angle_mode in [
            "range",
            "value",
        ], "angle mode should be in ['range', 'value', None]."
        self.scale = scale
        self.scale_mode = scale_mode
        self.angle = angle
        self.angle_mode = angle_mode
        self.fill_value = fill_value
        self.rotate_prob = rotate_prob
        self.auto_bound = auto_bound

    def get_value(self, value_list, value_mode):
        if not value_mode:
            return value_list
        elif value_mode == "range":
            low, high = value_list
            return np.random.rand() * (high - low) + low
        elif value_mode == "value":
            return np.random.choice(value_list)

    def apply(self, sample):
        if np.random.rand() > self.rotate_prob:
            return sample

        angle = self.get_value(self.angle, self.angle_mode)
        scale = self.get_value(self.scale, self.scale_mode)
        rotator = RRotate(scale, angle, self.fill_value, self.auto_bound)
        return rotator(sample)


class RResize(Transform):
    def __init__(self, target_size, keep_ratio=True, interp=cv2.INTER_LINEAR):
        """
        Resize image to target size. If `keep_ratio` is True,
        resize the image's long side to the maximum of `target_size`.
        If `keep_ratio` is False, resize the image to target size (h, w).

        Args:
            target_size (int|list): Image target size.
            keep_ratio (bool, optional): Whether or not to keep ratio. Default: True.
            interp (int, optional): Interpolation method. Default: cv2.INTER_LINEAR.
        """
        super(RResize, self).__init__()
        self.keep_ratio = keep_ratio
        self.interp = interp
        if not isinstance(target_size, (Integral, Sequence)):
            raise TypeError(
                "Type of `target_size` is invalid. Must be integer, list, or tuple, but now it is {}.".
                format(type(target_size)))
        if isinstance(target_size, Integral):
            target_size = [target_size, target_size]
        self.target_size = target_size

    def apply_im(self, image, scale):
        im_scale_x, im_scale_y = scale

        return cv2.resize(
            image,
            None,
            None,
            fx=im_scale_x,
            fy=im_scale_y,
            interpolation=self.interp)

    def apply_pts(self, pts, scale, size):
        im_scale_x, im_scale_y = scale
        resize_w, resize_h = size
        pts[:, 0::2] *= im_scale_x
        pts[:, 1::2] *= im_scale_y
        pts[:, 0::2] = np.clip(pts[:, 0::2], 0, resize_w)
        pts[:, 1::2] = np.clip(pts[:, 1::2], 0, resize_h)
        return pts

    def apply(self, sample):
        """Resize the image numpy."""
        im = sample["image"]
        if not isinstance(im, np.ndarray):
            raise TypeError("{}: image type is not numpy.".format(self))
        if len(im.shape) != 3:
            raise ValueError("{}: image is not 3-dimensional.".format(self))

        # apply image
        im_shape = im.shape
        if self.keep_ratio:

            im_size_min = np.min(im_shape[0:2])
            im_size_max = np.max(im_shape[0:2])

            target_size_min = np.min(self.target_size)
            target_size_max = np.max(self.target_size)

            im_scale = min(target_size_min / im_size_min,
                           target_size_max / im_size_max)

            resize_h = im_scale * float(im_shape[0])
            resize_w = im_scale * float(im_shape[1])

            im_scale_x = im_scale
            im_scale_y = im_scale
        else:
            resize_h, resize_w = self.target_size
            im_scale_y = resize_h / im_shape[0]
            im_scale_x = resize_w / im_shape[1]

        im = self.apply_im(sample["image"], [im_scale_x, im_scale_y])
        sample["image"] = im.astype(np.float32)
        sample["im_shape"] = np.asarray([resize_h, resize_w], dtype=np.float32)
        if "scale_factor" in sample:
            scale_factor = sample["scale_factor"]
            sample["scale_factor"] = np.asarray(
                [scale_factor[0] * im_scale_y, scale_factor[1] * im_scale_x],
                dtype=np.float32, )
        else:
            sample["scale_factor"] = np.asarray(
                [im_scale_y, im_scale_x], dtype=np.float32)

        # apply bbox
        if "gt_bbox" in sample and len(sample["gt_bbox"]) > 0:
            sample["gt_bbox"] = self.apply_pts(sample["gt_bbox"],
                                               [im_scale_x, im_scale_y],
                                               [resize_w, resize_h])

        # apply polygon
        if "gt_poly" in sample and len(sample["gt_poly"]) > 0:
            sample["gt_poly"] = self.apply_pts(sample["gt_poly"],
                                               [im_scale_x, im_scale_y],
                                               [resize_w, resize_h])

        return sample
