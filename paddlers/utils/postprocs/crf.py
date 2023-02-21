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
from skimage.color import gray2rgb
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_labels


def conditional_random_field(original_image: np.ndarray,
                             mask: np.ndarray) -> np.ndarray:
    """
    Conditional random field.

    The original article refers to
    Krhenbühl, Philipp, Koltun V. "Efficient Inference in Fully Connected CRFs with Gaussian Edge Potentials."
    (https://arxiv.org/abs/1210.5644v1).

    The implementation procedure refers to this repo: 
    https://github.com/apletea/Computer-Vision

    Args:
        original_image (np.ndarray): Original image. Shape is [H, W, 3]. 
        mask (np.ndarray): Mask to refine. Shape is [H, W].

    Returns:
        np.ndarray: Mask after CRF.
    """
    n_labels = len(np.unique(mask))
    mask3 = gray2rgb(mask)
    annotated_label = mask3[:, :, 0] + (mask3[:, :, 1] << 8) + (mask3[:, :, 2]
                                                                << 16)
    _, labels = np.unique(annotated_label, return_inverse=True)
    img_shape = original_image.shape
    d = dcrf.DenseCRF2D(img_shape[1], img_shape[0], n_labels)
    U = unary_from_labels(labels, n_labels, gt_prob=0.7, zero_unsure=False)
    d.setUnaryEnergy(U)
    d.addPairwiseGaussian(
        sxy=(3, 3),
        compat=3,
        kernel=dcrf.DIAG_KERNEL,
        normalization=dcrf.NORMALIZE_SYMMETRIC)
    Q = d.inference(10)
    MAP = np.argmax(Q, axis=0)
    MAP = MAP.reshape((img_shape[0], img_shape[1]))
    return MAP.astype("uint8")
