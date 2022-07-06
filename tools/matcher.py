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

import numpy as np
import cv2

from utils import Raster, raster2uint8, save_geotiff, timer

class MatchError(Exception):
    def __str__(self):
        return "Cannot match two images."


def _calcu_tf(im1, im2):
    orb = cv2.AKAZE_create()
    kp1, des1 = orb.detectAndCompute(im1, None)
    kp2, des2 = orb.detectAndCompute(im2, None)
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


@timer
def matching(im1_path, im2_path, im1_bands=[1, 2, 3], im2_bands=[1, 2, 3]):
    im1_ras = Raster(im1_path)
    im2_ras = Raster(im2_path)
    im1 = _get_match_img(im1_ras._src_data, im1_bands)
    im2 = _get_match_img(im2_ras._src_data, im2_bands)
    H = _calcu_tf(im1, im2)
    # test
    # im2_t = cv2.warpPerspective(im2, H, (im1.shape[1], im1.shape[0]))
    # cv2.imwrite("B_M.png", cv2.cvtColor(im2_t, cv2.COLOR_RGB2BGR))
    im2_arr_t = cv2.warpPerspective(im2_ras.getArray(), H,
                                    (im1_ras.width, im1_ras.height))
    save_path = im2_ras.path.replace(("." + im2_ras.ext_type), "_M.tif")
    save_geotiff(im2_arr_t, save_path, im1_ras.proj, im1_ras.geot, im1_ras.datatype)


parser = argparse.ArgumentParser(description="input parameters")
parser.add_argument("--im1_path", type=str, required=True, \
                    help="The path of time1 image (with geoinfo).")
parser.add_argument("--im2_path", type=str, required=True, \
                    help="The path of time2 image.")
parser.add_argument("--im1_bands", type=int, nargs="+", default=[1, 2, 3], \
                    help="The time1 image's band used for matching, RGB or monochrome, `[1, 2, 3]` is the default.")
parser.add_argument("--im2_bands", type=int, nargs="+", default=[1, 2, 3], \
                    help="The time2 image's band used for matching, RGB or monochrome, `[1, 2, 3]` is the default.")

if __name__ == "__main__":
    args = parser.parse_args()
    matching(args.im1_path, args.im2_path, args.im1_bands, args.im2_bands)
