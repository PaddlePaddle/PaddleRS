#!/usr/bin/env python

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

import argparse

import paddlers
import numpy as np
import cv2

from utils import Raster, raster2uint8, save_geotiff, time_it


class MatchError(Exception):
    def __str__(self):
        return "Cannot match the two images."


def _calcu_tf(image1, image2):
    orb = cv2.AKAZE_create()
    kp1, des1 = orb.detectAndCompute(image1, None)
    kp2, des2 = orb.detectAndCompute(image2, None)
    bf = cv2.BFMatcher()
    mathces = bf.knnMatch(des2, des1, k=2)
    good_matches = []
    for m, n in mathces:
        if m.distance < 0.75 * n.distance:
            good_matches.append([m])
    if len(good_matches) < 4:
        raise MatchError()
    src_automatic_points = np.float32([kp2[m[0].queryIdx].pt \
                                      for m in good_matches]).reshape(-1, 1, 2)
    den_automatic_points = np.float32([kp1[m[0].trainIdx].pt \
                                      for m in good_matches]).reshape(-1, 1, 2)
    H, _ = cv2.findHomography(src_automatic_points, den_automatic_points,
                              cv2.RANSAC, 5.0)
    return H


def _get_match_img(raster, bands):
    if len(bands) not in [1, 3]:
        raise ValueError("The lenght of bands must be 1 or 3.")
    band_array = []
    for b in bands:
        band_i = raster.GetRasterBand(b).ReadAsArray()
        band_array.append(band_i)
    if len(band_array) == 1:
        ima = raster2uint8(band_array[0])
    else:
        ima = raster2uint8(np.stack(band_array, axis=-1))
        ima = cv2.cvtColor(ima, cv2.COLOR_RGB2GRAY)
    return ima


@time_it
def match(image1_path,
          image2_path,
          save_path,
          image1_bands=[1, 2, 3],
          image2_bands=[1, 2, 3]):
    im1_ras = Raster(image1_path)
    im2_ras = Raster(image2_path)
    im1 = _get_match_img(im1_ras._src_data, image1_bands)
    im2 = _get_match_img(im2_ras._src_data, image2_bands)
    H = _calcu_tf(im1, im2)
    im2_arr_t = cv2.warpPerspective(im2_ras.getArray(), H,
                                    (im1_ras.width, im1_ras.height))
    save_geotiff(im2_arr_t, save_path, im1_ras.proj, im1_ras.geot,
                 im1_ras.datatype)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image1_path', type=str, required=True, \
                        help="Path of time1 image (with geoinfo).")
    parser.add_argument('--image2_path', type=str, required=True, \
                        help="Path of time2 image.")
    parser.add_argument('--save_path', type=str, required=True, \
                        help="Path to save matching result.")
    parser.add_argument('--image1_bands', type=int, nargs="+", default=[1, 2, 3], \
                        help="Bands of image1 to be used for matching, RGB or monochrome. The default value is [1, 2, 3].")
    parser.add_argument('--image2_bands', type=int, nargs="+", default=[1, 2, 3], \
                        help="Bands of image2 to be used for matching, RGB or monochrome. The default value is [1, 2, 3].")
    args = parser.parse_args()
    match(args.image1_path, args.image2_path, args.save_path, args.image1_bands,
          args.image2_bands)
