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

import codecs
import argparse

import cv2
import numpy as np
import geojson
from geojson import Polygon, Feature, FeatureCollection

from utils import Raster, Timer


def _gt_convert(x, y, geotf):
    x_geo = geotf[0] + x * geotf[1] + y * geotf[2]
    y_geo = geotf[3] + x * geotf[4] + y * geotf[5]
    return x_geo, y_geo


@Timer
def convert_data(mask_path, save_path, epsilon=0):
    raster = Raster(mask_path)
    img = raster.getArray()
    ext = save_path.split(".")[-1]
    if ext != "json" and ext != "geojson":
        raise ValueError("The ext of `save_path` must be `json` or `geojson`, not {}.".format(ext))
    geo_writer = codecs.open(save_path, "w", encoding="utf-8")
    clas = np.unique(img)
    cv2_v = (cv2.__version__.split(".")[0] == "3")
    feats = []
    if not isinstance(epsilon, (int, float)):
        epsilon = 0
    for iclas in range(1, len(clas)):
        tmp = np.zeros_like(img).astype("uint8")
        tmp[img == iclas] = 1
        # TODO: Detect internal and external contour
        results = cv2.findContours(tmp, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_TC89_KCOS)
        contours = results[1] if cv2_v else results[0]
        # hierarchys = results[2] if cv2_v else results[1]
        if len(contours) == 0:
            continue
        for contour in contours:
            contour = cv2.approxPolyDP(contour, epsilon, True)
            polys = []
            for point in contour:
                x, y = point[0]
                xg, yg = _gt_convert(x, y, raster.geot)
                polys.append((xg, yg))
            polys.append(polys[0])
            feat = Feature(
                geometry=Polygon([polys]), properties={"class": int(iclas)})
            feats.append(feat)
    gjs = FeatureCollection(feats)
    geo_writer.write(geojson.dumps(gjs))
    geo_writer.close()


parser = argparse.ArgumentParser(description="input parameters")
parser.add_argument("--mask_path", type=str, required=True, \
                    help="The path of mask tif.")
parser.add_argument("--save_path", type=str, required=True, \
                    help="The path to save the results, file suffix is `*.json/geojson`.")
parser.add_argument("--epsilon", type=float, default=0, \
                    help="The CV2 simplified parameters, `0` is the default.")

if __name__ == "__main__":
    args = parser.parse_args()
    convert_data(args.mask_path, args.save_path, args.epsilon)
