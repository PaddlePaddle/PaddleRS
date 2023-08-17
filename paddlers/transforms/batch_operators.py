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

import traceback
import random

import numpy as np
from paddle.io.dataloader.collate import default_collate_fn

from .operators import Transform, Resize, ResizeByShort, _Permute, interp_dict
from .box_utils import jaccard_overlap
from paddlers.utils import logging


class _IPermute(Transform):
    def __init__(self):
        super(_IPermute, self).__init__()

    def apply_im(self, image):
        image = np.moveaxis(image, 0, 2)
        return image


class BatchTransform(Transform):
    is_batch_transform = True


class BatchCompose(BatchTransform):
    def __init__(self, batch_transforms=None, collate_batch=True):
        super(BatchCompose, self).__init__()
        self.batch_transforms = batch_transforms
        self.collate_batch = collate_batch

    def __call__(self, samples):
        iperm_op = _IPermute()
        perm_status_list = []
        for i, sample in enumerate(samples):
            permuted = sample.pop('permuted', False)
            if permuted:
                # If a sample is permuted, we apply the inverse-permute
                # operator, such that it is possible to reuse non-batched data 
                # transformation operators later.
                samples[i] = iperm_op(sample)
            perm_status_list.append(permuted)

        if self.batch_transforms is not None:
            for op in self.batch_transforms:
                try:
                    samples = op(samples)
                except Exception as e:
                    stack_info = traceback.format_exc()
                    logging.warning("Fail to map batch transform [{}] "
                                    "with error: {} and stack:\n{}".format(
                                        op, e, str(stack_info)))
                    raise e

        # Recover permutation status
        perm_op = _Permute()
        for i, permuted in enumerate(perm_status_list):
            if permuted:
                samples[i] = perm_op(samples[i])

        extra_key = ['h', 'w', 'flipped', 'trans_info']

        for k in extra_key:
            for sample in samples:
                if k in sample:
                    sample.pop(k)

        if self.collate_batch:
            batch_data = default_collate_fn(samples)
        else:
            batch_data = {}
            for k in samples[0].keys():
                tmp_data = []
                for i in range(len(samples)):
                    tmp_data.append(samples[i][k])
                if not "gt_" in k and not "is_crowd" in k and not "difficult" in k:
                    # This if assumes that all elements in tmp_data has the same type.
                    if len(tmp_data) == 0 or not isinstance(tmp_data[0],
                                                            (str, bytes)):
                        tmp_data = np.stack(tmp_data, axis=0)
                batch_data[k] = tmp_data
        return batch_data


class BatchRandomResize(BatchTransform):
    """
    Resize a batch of inputs to random sizes.

    Attention: If `interp` is 'RANDOM', the interpolation method will be chosen randomly.

    Args:
        target_sizes (list[int] | list[list|tuple] | tuple[list|tuple]):
            Multiple target sizes, each of which should be an int or list/tuple of length 2.
        interp (str, optional): Interpolation method for resizing image(s). One of
            {'NEAREST', 'LINEAR', 'CUBIC', 'AREA', 'LANCZOS4', 'RANDOM'}.
            Defaults to 'LINEAR'.
    Raises:
        TypeError: Invalid type of `target_size`.
        ValueError: Invalid interpolation method.

    See Also:
        RandomResize: Resize input to random sizes.
    """

    def __init__(self, target_sizes, interp="NEAREST"):
        super(BatchRandomResize, self).__init__()
        if not (interp == "RANDOM" or interp in interp_dict):
            raise ValueError("`interp` should be one of {}".format(
                interp_dict.keys()))
        self.interp = interp
        assert isinstance(target_sizes, list), "`target_size` must be a list."
        for i, item in enumerate(target_sizes):
            if isinstance(item, int):
                target_sizes[i] = (item, item)
        self.target_size = target_sizes

    def __call__(self, samples):
        height, width = random.choice(self.target_size)
        resizer = Resize((height, width), interp=self.interp)
        samples = resizer(samples)

        return samples


class BatchRandomResizeByShort(BatchTransform):
    """
    Resize a batch of inputs to random sizes while keeping the aspect ratio.

    Attention: If `interp` is 'RANDOM', the interpolation method will be chosen randomly.

    Args:
        short_sizes (list[int] | tuple[int]): Target sizes of the shorter side of
            the image(s).
        max_size (int, optional): Upper bound of longer side of the image(s).
            If `max_size` is -1, no upper bound will be applied. Defaults to -1.
        interp (str, optional): Interpolation method for resizing image(s). One of
            {'NEAREST', 'LINEAR', 'CUBIC', 'AREA', 'LANCZOS4', 'RANDOM'}.
            Defaults to 'LINEAR'.

    Raises:
        TypeError: Invalid type of `target_size`.
        ValueError: Invalid interpolation method.

    See Also:
        RandomResizeByShort: Resize input to random sizes while keeping the aspect
            ratio.
    """

    def __init__(self, short_sizes, max_size=-1, interp="NEAREST"):
        super(BatchRandomResizeByShort, self).__init__()
        if not (interp == "RANDOM" or interp in interp_dict):
            raise ValueError("`interp` should be one of {}".format(
                interp_dict.keys()))
        self.interp = interp
        assert isinstance(short_sizes, list), "`short_sizes` must be a list."

        self.short_sizes = short_sizes
        self.max_size = max_size

    def __call__(self, samples):
        short_size = random.choice(self.short_sizes)
        resizer = ResizeByShort(
            short_size=short_size, max_size=self.max_size, interp=self.interp)

        samples = resizer(samples)

        return samples


class BatchNormalizeImage(BatchTransform):
    def __init__(
            self,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            is_scale=True,
            norm_type="mean_std", ):
        """
        Args:
            mean (list, optional): Pixel mean values. Default: [0.485, 0.456, 0.406].
            std (list, optional): Pixel variance valus. Default: [0.485, 0.456, 0.406].
            is_scale (bool, optional): Whether to scale the pixel values to [0, 1]. 
                Default: True.
            norm_type (str, optional): One of ['mean_std', 'none']. Default: 'mean_std'.
        """
        super(BatchNormalizeImage, self).__init__()
        self.mean = mean
        self.std = std
        self.is_scale = is_scale
        self.norm_type = norm_type
        if not (isinstance(self.mean, list) and isinstance(self.std, list) and
                isinstance(self.is_scale, bool) and
                self.norm_type in ["mean_std", "none"]):
            raise TypeError("{}: input type is invalid.".format(self))
        from functools import reduce

        if reduce(lambda x, y: x * y, self.std) == 0:
            raise ValueError("{}: std is invalid!".format(self))

    def apply(self, sample):
        """Normalize the image.
        Operators:
            1.(optional) Scale the pixel to [0,1]
            2.(optional) Each pixel minus mean and is divided by std
        """
        im = sample["image"]

        im = im.astype(np.float32, copy=False)
        if self.is_scale:
            scale = 1.0 / 255.0
            im *= scale

        if self.norm_type == "mean_std":
            mean = np.array(self.mean)[np.newaxis, np.newaxis, :]
            std = np.array(self.std)[np.newaxis, np.newaxis, :]
            im -= mean
            im /= std

        sample["image"] = im

        if "pre_image" in sample:
            pre_im = sample["pre_image"]
            pre_im = pre_im.astype(np.float32, copy=False)
            if self.is_scale:
                scale = 1.0 / 255.0
                pre_im *= scale

            if self.norm_type == "mean_std":
                mean = np.array(self.mean)[np.newaxis, np.newaxis, :]
                std = np.array(self.std)[np.newaxis, np.newaxis, :]
                pre_im -= mean
                pre_im /= std
            sample["pre_image"] = pre_im

        return sample


class BatchPadRGT(BatchTransform):
    """
    Pad `gt_class`, `gt_bbox`, `gt_score` with zero value.

    Args:
        return_gt_mask (bool, optional): If True, return `pad_gt_mask`. In
            `pad_gt_mask`, 1 means there is a bbox while 0 means there is none.
            Default: True.
    """

    def __init__(self, return_gt_mask=True):
        super(BatchPadRGT, self).__init__()
        self.return_gt_mask = return_gt_mask

    def pad_field(self, sample, field, num_gt):
        name, shape, dtype = field
        if name in sample:
            pad_v = np.zeros(shape, dtype=dtype)
            if num_gt > 0:
                pad_v[:num_gt] = sample[name]
            sample[name] = pad_v

    def __call__(self, samples):
        num_max_boxes = max([len(s["gt_bbox"]) for s in samples])
        for sample in samples:
            if self.return_gt_mask:
                sample["pad_gt_mask"] = np.zeros(
                    (num_max_boxes, 1), dtype=np.float32)
            if num_max_boxes == 0:
                continue

            num_gt = len(sample["gt_bbox"])
            pad_gt_class = np.zeros((num_max_boxes, 1), dtype=np.int32)
            pad_gt_bbox = np.zeros((num_max_boxes, 4), dtype=np.float32)
            if num_gt > 0:
                pad_gt_class[:num_gt] = sample["gt_class"]
                pad_gt_bbox[:num_gt] = sample["gt_bbox"]
            sample["gt_class"] = pad_gt_class
            sample["gt_bbox"] = pad_gt_bbox
            # pad_gt_mask
            if "pad_gt_mask" in sample:
                sample["pad_gt_mask"][:num_gt] = 1
            # gt_score
            names = ["gt_score", "is_crowd", "difficult", "gt_poly", "gt_rbox"]
            dims = [1, 1, 1, 8, 5]
            dtypes = [np.float32, np.int32, np.int32, np.float32, np.float32]

            for name, dim, dtype in zip(names, dims, dtypes):
                self.pad_field(sample, [name, (num_max_boxes, dim), dtype],
                               num_gt)

        return samples


class _BatchPad(BatchTransform):
    def __init__(self, pad_to_stride=0):
        super(_BatchPad, self).__init__()
        self.pad_to_stride = pad_to_stride

    def __call__(self, samples):
        coarsest_stride = self.pad_to_stride
        max_shape = np.array([data["image"].shape for data in samples]).max(
            axis=0)
        if coarsest_stride > 0:
            max_shape[0] = int(
                np.ceil(max_shape[0] / coarsest_stride) * coarsest_stride)
            max_shape[1] = int(
                np.ceil(max_shape[1] / coarsest_stride) * coarsest_stride)
        for data in samples:
            im = data["image"]
            im_h, im_w, im_c = im.shape[:]
            padding_im = np.zeros(
                (max_shape[0], max_shape[1], im_c), dtype=np.float32)
            padding_im[:im_h, :im_w, :] = im
            data["image"] = padding_im

        return samples


class _Gt2YoloTarget(BatchTransform):
    """
    Generate YOLOv3 targets from ground-truth data, this operator is only used in
        fine grained YOLOv3 loss mode.
    """

    def __init__(self,
                 anchors,
                 anchor_masks,
                 downsample_ratios,
                 num_classes=80,
                 iou_thresh=1.0):
        super(_Gt2YoloTarget, self).__init__()
        self.anchors = anchors
        self.anchor_masks = anchor_masks
        self.downsample_ratios = downsample_ratios
        self.num_classes = num_classes
        self.iou_thresh = iou_thresh

    def __call__(self, samples, context=None):
        assert len(self.anchor_masks) == len(
            self.downsample_ratios
        ), "`anchor_masks` and `downsample_ratios` should have same length."

        h, w = samples[0]["image"].shape[:2]
        an_hw = np.array(self.anchors) / np.array([[w, h]])
        for sample in samples:
            gt_bbox = sample["gt_bbox"]
            gt_class = sample["gt_class"]
            if "gt_score" not in sample:
                sample["gt_score"] = np.ones(
                    (gt_bbox.shape[0], 1), dtype=np.float32)
            gt_score = sample["gt_score"]
            for i, (
                    mask, downsample_ratio
            ) in enumerate(zip(self.anchor_masks, self.downsample_ratios)):
                grid_h = int(h / downsample_ratio)
                grid_w = int(w / downsample_ratio)
                target = np.zeros(
                    (len(mask), 6 + self.num_classes, grid_h, grid_w),
                    dtype=np.float32)
                for b in range(gt_bbox.shape[0]):
                    gx, gy, gw, gh = gt_bbox[b, :]
                    cls = gt_class[b]
                    score = gt_score[b]
                    if gw <= 0.0 or gh <= 0.0 or score <= 0.0:
                        continue

                    # Find best matched anchor index
                    best_iou = 0.0
                    best_idx = -1
                    for an_idx in range(an_hw.shape[0]):
                        iou = jaccard_overlap(
                            [0.0, 0.0, gw, gh],
                            [0.0, 0.0, an_hw[an_idx, 0], an_hw[an_idx, 1]], )
                        if iou > best_iou:
                            best_iou = iou
                            best_idx = an_idx

                    gi = int(gx * grid_w)
                    gj = int(gy * grid_h)

                    # gtbox should be regressed in this layer if best matched
                    # anchor index is in the anchor mask of this layer.
                    if best_idx in mask:
                        best_n = mask.index(best_idx)

                        # x, y, w, h, scale
                        target[best_n, 0, gj, gi] = gx * grid_w - gi
                        target[best_n, 1, gj, gi] = gy * grid_h - gj
                        target[best_n, 2, gj, gi] = np.log(
                            gw * w / self.anchors[best_idx][0])
                        target[best_n, 3, gj, gi] = np.log(
                            gh * h / self.anchors[best_idx][1])
                        target[best_n, 4, gj, gi] = 2.0 - gw * gh

                        # Record gt_score
                        target[best_n, 5, gj, gi] = score

                        # Do classification
                        target[best_n, 6 + cls, gj, gi] = 1.0

                    # For non-matched anchors, calculate the target if the iou
                    # between anchor and gt is larger than iou_thresh.
                    if self.iou_thresh < 1:
                        for idx, mask_i in enumerate(mask):
                            if mask_i == best_idx:
                                continue
                            iou = jaccard_overlap(
                                [0.0, 0.0, gw, gh],
                                [0.0, 0.0, an_hw[mask_i, 0], an_hw[mask_i, 1]],
                            )
                            if iou > self.iou_thresh and target[idx, 5, gj,
                                                                gi] == 0.0:
                                # x, y, w, h, scale
                                target[idx, 0, gj, gi] = gx * grid_w - gi
                                target[idx, 1, gj, gi] = gy * grid_h - gj
                                target[idx, 2, gj, gi] = np.log(
                                    gw * w / self.anchors[mask_i][0])
                                target[idx, 3, gj, gi] = np.log(
                                    gh * h / self.anchors[mask_i][1])
                                target[idx, 4, gj, gi] = 2.0 - gw * gh

                                # Record gt_score
                                target[idx, 5, gj, gi] = score

                                # Do classification
                                target[idx, 5 + cls, gj, gi] = 1.0
                sample["target{}".format(i)] = target

            # Remove useless gt_class and gt_score items after target has been calculated.
            sample.pop("gt_class")
            sample.pop("gt_score")

        return samples
