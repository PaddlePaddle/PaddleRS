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

import numpy as np
import cv2


def prepro_mask(mask: np.ndarray, area_threshold: int = 32) -> np.ndarray:
    mask_shape = mask.shape
    if len(mask_shape) != 2:
        mask = mask[..., 0]
    mask = mask.astype("uint8")
    mask = _del_small_connection(mask, area_threshold)
    class_num = len(np.unique(mask))
    if class_num != 2:
        _, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY |
                                cv2.THRESH_OTSU)
    mask = np.clip(mask, 0, 1).astype("uint8")  # 0-255 / 0-1 -> 0-1
    return mask


def calc_distance(p1: np.ndarray, p2: np.ndarray) -> float:
    return float(np.sqrt(np.sum(np.power((p1[0] - p2[0]), 2))))


def _del_small_connection(pred: np.ndarray, threshold: int = 32) -> np.ndarray:
    result = np.zeros_like(pred)
    contours, reals = cv2.findContours(pred, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for contour, real in zip(contours, reals[0]):
        if real[-1] == -1:
            if cv2.contourArea(contour) > threshold:
                cv2.fillPoly(result, [contour], (1))
        else:
            cv2.fillPoly(result, [contour], (0))
    return result.astype("uint8")
