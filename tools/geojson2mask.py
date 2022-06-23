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
import cv2
import numpy as np
import argparse
import geojson
from tqdm import tqdm
from utils import Raster, save_geotiff, Timer


def _gt_convert(x_geo, y_geo, geotf):
    a = np.array([[geotf[1], geotf[2]], [geotf[4], geotf[5]]])
    b = np.array([x_geo - geotf[0], y_geo - geotf[3]])
    return np.round(np.linalg.solve(a, b)).tolist()  # 解一元二次方程


@Timer
def convert_data(image_path, geojson_path):
    raster = Raster(image_path)
    tmp_img = np.zeros((raster.height, raster.width), dtype=np.int32)
    geo_reader = codecs.open(geojson_path, "r", encoding="utf-8")
    feats = geojson.loads(geo_reader.read())["features"]  # 所有图像块
    for feat in tqdm(feats):
        geo = feat["geometry"]
        if geo["type"] == "Polygon":  # 多边形
            geo_points = geo["coordinates"][0]
        elif geo["type"] == "MultiPolygon":  # 多面
            geo_points = geo["coordinates"][0][0]
        else:
            raise TypeError("Geometry type must be `Polygon` or `MultiPolygon`, not {}.".format(geo["type"]))
        xy_points = np.array([
            _gt_convert(point[0], point[1], raster.geot)
            for point in geo_points
        ]).astype(np.int32)
        # TODO: Label category
        cv2.fillPoly(tmp_img, [xy_points], 1)  # 多边形填充
    ext = "." + geojson_path.split(".")[-1]
    save_geotiff(tmp_img, geojson_path.replace(ext, ".tif"), raster.proj, raster.geot)


parser = argparse.ArgumentParser(description="input parameters")
parser.add_argument("--image_path", type=str, required=True, \
                    help="The path of original image.")
parser.add_argument("--geojson_path", type=str, required=True, \
                    help="The path of geojson.")


if __name__ == "__main__":
    args = parser.parse_args()
    convert_data(args.image_path, args.geojson_path)
