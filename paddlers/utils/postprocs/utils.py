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

from typing import Union, Callable, Dict, Any

import cv2
import numpy as np
import paddle


def calc_distance(p1: np.ndarray, p2: np.ndarray) -> float:
    return float(np.sqrt(np.sum(np.power((p1[0] - p2[0]), 2))))


def prepro_mask(input: Union[paddle.Tensor, np.ndarray]) -> np.ndarray:
    """
    Standardized mask.

    Args:
        input (Union[paddle.Tensor, np.ndarray]): Mask to refine, or user's mask.

    Returns:
        np.ndarray: Standard mask.
    """
    input_shape = input.shape
    if isinstance(input, paddle.Tensor):
        if len(input_shape) == 4:
            mask = paddle.argmax(input, axis=1).squeeze_().numpy()
        else:
            raise ValueError("Invalid tensor, shape must be 4, not " + str(
                input_shape) + ".")
    else:
        if len(input_shape) == 3:
            mask = input[..., 0]
        elif len(input_shape) == 2:
            mask = input
        else:
            raise ValueError("Invalid ndarray, shape must be 2 or 3, not " +
                             str(input_shape) + ".")
        mask = mask.astype("uint8")
        class_mask = np.unique(mask)
        if len(class_mask) == 2:
            mask = np.clip(mask, 0, 1)  # 0-255 / 0-1 -> 0-1
        else:
            if (max(class_mask) > (len(class_mask - 1))):
                _, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY |
                                        cv2.THRESH_OTSU)
                mask = np.clip(mask, 0, 1)
    return mask.astype("uint8")


def del_small_connection(mask: np.ndarray, threshold: int=32) -> np.ndarray:
    """
    Delete the connected region whose pixel area is less than the threshold from mask.

    Args:
        mask (np.ndarray): Mask to refine. Shape is [H, W] and values are 0 or 1.
        threshold (int, optional): Threshold of deleted area. Default is 32.

    Returns:
        np.ndarray: Mask after deleted samll connection.
    """
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
        mask (np.ndarray): Mask to refine. Shape is [H, W] and values are 0 or 1.
        threshold (int, optional): Threshold of filled area. Default is 32.

    Returns:
        np.ndarray: Mask after deleted samll connection.
    """
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


def morphological_operation(mask: np.ndarray,
                            ops: str="open",
                            k_size: int=3,
                            iterations: int=1) -> np.ndarray:
    """
    Morphological operation.
    Open: It is used to separate objects and eliminate small areas.
    Close: It is used to eliminating small holes.
    Erode: It is used to refine goals.
    Dilate: It is used to Coarse goals.

    Args:
        mask (np.ndarray): Mask to refine. Shape is [H, W].
        ops (str): . Defaults to "open".
        k_size (int, optional): Size of the structuring element. Defaults to 3.
        iterations (int, optional): Number of times erosion and dilation are applied. Defaults to 1.

    Returns:
        np.ndarray: Morphologically processed mask.
    """
    kv = {
        "open": cv2.MORPH_OPEN,
        "close": cv2.MORPH_CLOSE,
        "erode": cv2.MORPH_ERODE,
        "dilate": cv2.MORPH_DILATE,
    }
    if ops.lower() not in kv.keys():
        raise ValueError("Invalid ops: " + ops +
                         ", `ops` must be `open/close/erode/dilate`.")
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
    opened = cv2.morphologyEx(
        mask, kv[ops.lower()], kernel, iterations=iterations)
    return opened.astype("uint8")


def deal_one_class(mask: np.ndarray,
                   class_index: int,
                   func: Callable,
                   **kwargs: Dict[str, Any]) -> np.ndarray:
    """
    Only a single category is processed. 

    Args:
        mask (np.ndarray): Mask to refine. Shape is [H, W].
        class_index (int): Index of class of need processed.
        func (Callable): Function of processed.

    Returns:
        np.ndarray: Processed Mask.
    """
    btmp = (mask == class_index).astype("uint8")
    res = func(btmp, **kwargs)
    res *= class_index
    return np.where(btmp == 0, mask, res).astype("uint8")
