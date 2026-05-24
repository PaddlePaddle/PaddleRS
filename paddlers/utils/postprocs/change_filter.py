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

from typing import Dict, Optional, Any

import numpy as np

from paddlers.transforms.operators import AppendIndex


def change_detection_filter(mask: np.ndarray,
                            t1: np.ndarray,
                            t2: np.ndarray,
                            threshold1: float,
                            threshold2: float,
                            index_type: str="NDVI",
                            band_indices: Optional[Dict]=None,
                            satellite: Optional[str]=None,
                            **kwargs: Dict[str, Any]) -> np.ndarray:
    """
    Remote sensing index filter. It is a postprocessing method for change detection tasks.

    E.g. Filter plant seasonal variations in non-urban scenes
    1. Calculate NDVI of the two images separately
    2. Obtain vegetation mask by threshold filter
    3. Take the intersection of the two vegetation masks, called veg_mask
    4. Filter mask through veg_mask

    Args:
        mask (np.ndarray): Change mask predicted by a change detection model. Shape is [H, W].
        t1 (np.ndarray): Original image of time 1.
        t2 (np.ndarray): Original image of time 2.
        threshold1 (float): Threshold of time 1.
        threshold2 (float): Threshold of time 2.
        
        For other arguments please refer to the data transformation operator `AppendIndex`
        (paddlers/transforms/operators.py)

    Returns:
        np.ndarray: Filtered mask.
    """
    index_calculator = AppendIndex(index_type, band_indices, satellite,
                                   **kwargs)
    index1 = index_calculator._compute_index(t1)
    index2 = index_calculator._compute_index(t2)
    imask1 = (index1 > threshold1).astype("uint8")
    imask2 = (index2 > threshold2).astype("uint8")
    imask = (imask1 + imask2 != 2).astype("uint8")
    return mask * imask
