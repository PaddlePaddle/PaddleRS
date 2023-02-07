# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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


def prepro_mask(mask: np.ndarray) -> np.ndarray:
    mask_shape = mask.shape
    if len(mask_shape) != 2:
        mask = mask[..., 0]
    mask = mask.astype("uint8")
    class_mask = np.unique(mask)
    if len(class_mask) == 3 and 255 in class_mask:
        _, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY |
                                cv2.THRESH_OTSU)
    elif len(class_mask) == 2:
        mask = np.clip(mask, 0, 1).astype("uint8")  # 0-255 / 0-1 -> 0-1
    return mask


def calc_distance(p1: np.ndarray, p2: np.ndarray) -> float:
    return float(np.sqrt(np.sum(np.power((p1[0] - p2[0]), 2))))


def del_small_connection(mask: np.ndarray, threshold: int=32) -> np.ndarray:
    """
    Delete the connected region whose pixel area is less than the threshold from mask.

    Args:
        mask (np.ndarray): Mask of infer.
        threshold (int, optional): Threshold of deleted area. Default is 32.

    Returns:
        np.ndarray: Mask after deleted samll connection.
    """
    mask = prepro_mask(mask)
    result = np.zeros_like(mask)
    contours, reals = cv2.findContours(mask, cv2.RETR_TREE,
                                       cv2.CHAIN_APPROX_NONE)
    for contour, real in zip(contours, reals[0]):
        if real[-1] == -1:
            if cv2.contourArea(contour) > threshold:
                cv2.fillPoly(result, [contour], (1))
        else:
            cv2.fillPoly(result, [contour], (0))
    return result.astype("uint8")


def fill_small_holes(mask: np.ndarray, threshold: int=32) -> np.ndarray:
    """
    Fill the holed region whose pixel area is less than the threshold from mask.

    Args:
        mask (np.ndarray): Mask of infer.
        threshold (int, optional): Threshold of filled area. Default is 32.

    Returns:
        np.ndarray: Mask after deleted samll connection.
    """
    mask = prepro_mask(mask)
    result = np.zeros_like(mask)
    contours, reals = cv2.findContours(mask, cv2.RETR_TREE,
                                       cv2.CHAIN_APPROX_NONE)
    for contour, real in zip(contours, reals[0]):
        # Fill father
        if real[-1] == -1:
            cv2.fillPoly(result, [contour], (1))
        # Fill children whose area less than threshold
        elif real[-1] != -1 and cv2.contourArea(contour) < threshold:
            cv2.fillPoly(result, [contour], (1))
        else:
            cv2.fillPoly(result, [contour], (0))
    return result.astype("uint8")


def open(mask: np.ndarray, k_size: int=3, iterations: int=1) -> np.ndarray:
    """
    Open operation. It support to separate objects and eliminate small areas.

    Args:
        mask (np.ndarray): Mask of infer.
        k_size (int, optional): Size of the structuring element. Defaults to 3.
        iterations (int, optional): Number of times erosion and dilation are applied. Defaults to 1.

    Returns:
        np.ndarray: Mask after open operation.
    """
    mask = prepro_mask(mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
    open = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=iterations)
    return open


def close(mask: np.ndarray, k_size: int=3, iterations: int=1) -> np.ndarray:
    """
    Close operation. It support to eliminating small holes.

    Args:
        mask (np.ndarray): Mask of infer.
        k_size (int, optional): Size of the structuring element. Defaults to 3.
        iterations (int, optional): Number of times erosion and dilation are applied. Defaults to 1.

    Returns:
        np.ndarray: Mask after close operation.
    """
    mask = prepro_mask(mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
    close = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    return close
