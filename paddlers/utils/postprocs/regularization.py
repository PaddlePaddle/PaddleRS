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

import cv2
import numpy as np
from .utils import *

S = 32


def building_regularization(mask: np.ndarray) -> np.ndarray:
    """
    Translate the mask of building into structured mask.

    The original article refers to
    Wei S, Ji S, Lu M. "Toward Automatic Building Footprint Delineation From Aerial Images Using CNN and Regularization."
    (https://ieeexplore.ieee.org/document/8933116).
        

    Args:
        mask (np.ndarray): The mask of building.

    Returns:
        np.ndarray: The mask of building after regularized.
    """
    # check
    shapes = mask.shape
    slens = len(shapes)
    if slens != 2:
        mask = (mask[..., 0])
    clases = np.unique(mask)
    clens = len(clases)
    if clens != 2:
        raise ValueError(
            "The number of categories in mask must be 2, not {}.".format(clens))
    mask = np.clip(mask, 0, 1).astype("uint8")  # 0-255 / 0-1 -> 0-1
    # find contours
    contours, hierarchys = cv2.findContours(mask, cv2.RETR_TREE,
                                            cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("There are no contours.")
    # coarse
    hconts = []
    for contour, hierarchy in zip(contours, hierarchys[0]):
        area = cv2.contourArea(contour)
        if area >= S:  #  remove polygons whose area is below a threshold S
            epsilon = 0.005 * cv2.arcLength(contour, True)
            contour = cv2.approxPolyDP(contour, epsilon, True)  # DP
            hconts.append((contour, _get_priority(hierarchy)))
    return _fill(mask, hconts)  # fill


def _get_priority(hierarchy) -> int:
    print(type(hierarchy))
    if hierarchy[3] < 0:
        return 1
    if hierarchy[2] < 0:
        return 2
    return 3


def _fill(img, hconts) -> np.ndarray:
    result = np.zeros_like(img)
    sorted(hconts, key=lambda x: x[1])
    for contour, priority in hconts:
        if priority == 2:
            cv2.fillPoly(result, [contour], (0, 0, 0))
        else:
            cv2.fillPoly(result, [contour], (255, 255, 255))
    return result
