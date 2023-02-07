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

import cv2
import numpy as np


def markov_random_field(original_image: np.ndarray,
                        mask: np.ndarray,
                        max_iter: int=2) -> np.ndarray:
    """
    Markov random field.

    Args:
        original_image (np.ndarray): Original image. Shape is [H, W, 3]. 
        mask (np.ndarray): Mask of infer. Shape is [H, W].
        max_iter (int, optional): Maximum number of iterations. Defaults to 2.

    Returns:
        np.ndarray: Mask after MRF.
    """
    img = original_image.astype("double")
    classes = sorted(np.unique(mask).tolist())
    cluster_num = len(classes)
    zlab = np.zeros_like(mask)
    for idx, pix in enumerate(classes, start=1):
        zlab[mask == pix] = idx
    mask = zlab.astype('int64')
    res = _MRF(img, mask, max_iter, cluster_num)
    return res.astype("uint8")


def _MRF(img, label, max_iter, cluster_num):
    f_u = np.array([0, 1, 0, 0, 0, 0, 0, 0, 0]).reshape(3, 3)
    f_d = np.array([0, 0, 0, 0, 0, 0, 0, 1, 0]).reshape(3, 3)
    f_l = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0]).reshape(3, 3)
    f_r = np.array([0, 0, 0, 0, 0, 1, 0, 0, 0]).reshape(3, 3)
    f_ul = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0]).reshape(3, 3)
    f_ur = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0]).reshape(3, 3)
    f_dl = np.array([0, 0, 0, 0, 0, 0, 1, 0, 0]).reshape(3, 3)
    f_dr = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1]).reshape(3, 3)
    iter = 0
    while iter < max_iter:
        iter = iter + 1
        # print(iter)
        label_u = cv2.filter2D(np.array(label, dtype=np.uint8), -1, f_u)
        label_d = cv2.filter2D(np.array(label, dtype=np.uint8), -1, f_d)
        label_l = cv2.filter2D(np.array(label, dtype=np.uint8), -1, f_l)
        label_r = cv2.filter2D(np.array(label, dtype=np.uint8), -1, f_r)
        label_ul = cv2.filter2D(np.array(label, dtype=np.uint8), -1, f_ul)
        label_ur = cv2.filter2D(np.array(label, dtype=np.uint8), -1, f_ur)
        label_dl = cv2.filter2D(np.array(label, dtype=np.uint8), -1, f_dl)
        label_dr = cv2.filter2D(np.array(label, dtype=np.uint8), -1, f_dr)
        m, n = label.shape
        p_c = np.zeros((cluster_num, m, n))
        for i in range(cluster_num):
            label_i = (i + 1) * np.ones((m, n))
            u_T = 1 * np.logical_not(label_i - label_u)
            d_T = 1 * np.logical_not(label_i - label_d)
            l_T = 1 * np.logical_not(label_i - label_l)
            r_T = 1 * np.logical_not(label_i - label_r)
            ul_T = 1 * np.logical_not(label_i - label_ul)
            ur_T = 1 * np.logical_not(label_i - label_ur)
            dl_T = 1 * np.logical_not(label_i - label_dl)
            dr_T = 1 * np.logical_not(label_i - label_dr)
            temp = u_T + d_T + l_T + r_T + ul_T + ur_T + dl_T + dr_T
            p_c[i, :] = (1.0 / 8) * temp
        p_c[p_c == 0] = 0.0001
        mu = np.zeros((1, cluster_num))
        sigma = np.zeros((1, cluster_num))
        for i in range(cluster_num):
            index = np.where(label == (i + 1))
            data_c = img[index]
            mu[0, i] = np.mean(data_c)
            sigma[0, i] = np.var(data_c)
        p_sc = np.zeros((cluster_num, m, n))
        one_a = np.ones((m, n))
        for j in range(cluster_num):
            MU = mu[0, j] * one_a
            p_sc[j, :] = (1. / np.sqrt(2. * np.pi * sigma[0, j])) * np.exp(
                -1. * ((img - MU)**2) / (2 * sigma[0, j]))
        X_out = np.log(p_c) + np.log(p_sc)
        label_c = X_out.reshape(cluster_num, m * n)
        label_c_t = label_c.T
        label_m = np.argmax(label_c_t, axis=1)
        label_m = label_m + np.ones(label_m.shape)
        label = label_m.reshape(m, n)
    return label
