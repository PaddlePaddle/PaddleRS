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

import os
import codecs
import argparse

import paddlers
import numpy as np
import cv2
import geojson
from tqdm import tqdm

from utils import Raster, save_geotiff, translate_vector, time_it


def _gt_convert(x_geo, y_geo, geotf):
    a = np.array([[geotf[1], geotf[2]], [geotf[4], geotf[5]]])
    b = np.array([x_geo - geotf[0], y_geo - geotf[3]])
    return np.round(np.linalg.solve(a,
                                    b)).tolist()  # Solve a quadratic equation


@time_it
# TODO: update for vector2raster
def convert_data(src_img_path, geojson_path, save_path):
    raster = Raster(src_img_path)
    tmp_img = np.zeros((raster.height, raster.width), dtype=np.int32)
    # vector to EPSG from raster
    temp_geojson_path = translate_vector(geojson_path, raster.proj)
    geo_reader = codecs.open(temp_geojson_path, "r", encoding="utf-8")
    feats = geojson.loads(geo_reader.read())["features"]  # All image patches
    geo_reader.close()
    for feat in tqdm(feats):
        geo = feat["geometry"]
        if geo["type"] == "Polygon":
            geo_points = geo["coordinates"][0]
        elif geo["type"] == "MultiPolygon":
            geo_points = geo["coordinates"][0][0]
        else:
            raise TypeError(
                "Geometry type must be 'Polygon' or 'MultiPolygon', not {}.".
                format(geo["type"]))
        xy_points = np.array([
            _gt_convert(point[0], point[1], raster.geot) for point in geo_points
        ]).astype(np.int32)
        # TODO: Label category
        cv2.fillPoly(tmp_img, [xy_points], 1)  # Fill with polygons
    save_geotiff(tmp_img, save_path, raster.proj, raster.geot)
    os.remove(temp_geojson_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_img_path", type=str, required=True, \
                        help="Path of the original image.")
    parser.add_argument("--geojson_path", type=str, required=True, \
                        help="Path of the GeoJSON file (the coordinate system is WGS84).")
    parser.add_argument("--save_path", type=str, required=True, \
                        help="Path to store the mask data.")
    args = parser.parse_args()
    convert_data(args.src_img_path, args.geojson_path, args.save_path)
