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
import os.path as osp
import argparse

import paddlers
import numpy as np
from PIL import Image
try:
    from osgeo import gdal, ogr, osr
except ImportError:
    import gdal
    import ogr
    import osr

from utils import Raster, save_geotiff, time_it


def _mask2tif(mask_path, tmp_path, proj, geot):
    dst_ds = save_geotiff(
        np.asarray(Image.open(mask_path)), tmp_path, proj, geot,
        gdal.GDT_UInt16, False)
    return dst_ds


def _polygonize_raster(mask_path, vec_save_path, proj, geot, ignore_index, ext):
    if proj is None or geot is None:
        tmp_path = None
        ds = gdal.Open(mask_path)
    else:
        tmp_path = vec_save_path.replace("." + ext, ".tif")
        ds = _mask2tif(mask_path, tmp_path, proj, geot)
    srcband = ds.GetRasterBand(1)
    maskband = srcband.GetMaskBand()
    gdal.SetConfigOption("GDAL_FILENAME_IS_UTF8", "YES")
    gdal.SetConfigOption("SHAPE_ENCODING", "UTF-8")
    ogr.RegisterAll()
    drv = ogr.GetDriverByName("ESRI Shapefile" if ext == "shp" else "GeoJSON")
    if osp.exists(vec_save_path):
        os.remove(vec_save_path)
    dst_ds = drv.CreateDataSource(vec_save_path)
    prosrs = osr.SpatialReference(wkt=ds.GetProjection())
    dst_layer = dst_ds.CreateLayer(
        "POLYGON", geom_type=ogr.wkbPolygon, srs=prosrs)
    dst_fieldname = "CLAS"
    fd = ogr.FieldDefn(dst_fieldname, ogr.OFTInteger)
    dst_layer.CreateField(fd)
    gdal.Polygonize(srcband, maskband, dst_layer, 0, [])
    # TODO: temporary: delete ignored values
    dst_ds.Destroy()
    ds = None
    vec_ds = drv.Open(vec_save_path, 1)
    lyr = vec_ds.GetLayer()
    lyr.SetAttributeFilter("{} = '{}'".format(dst_fieldname, str(ignore_index)))
    for holes in lyr:
        lyr.DeleteFeature(holes.GetFID())
    vec_ds.Destroy()
    if tmp_path is not None:
        os.remove(tmp_path)


@time_it
def mask2shape(src_img_path, mask_path, save_path, ignore_index=255):
    vec_ext = save_path.split(".")[-1].lower()
    if vec_ext not in ["json", "geojson", "shp"]:
        raise ValueError(
            "The extension of `save_path` must be 'json/geojson' or 'shp', not {}.".
            format(vec_ext))
    ras_ext = src_img_path.split(".")[-1].lower()
    if osp.exists(
            src_img_path) and ras_ext in ["tif", "tiff", "geotiff", "img"]:
        src = Raster(src_img_path)
        _polygonize_raster(mask_path, save_path, src.proj, src.geot,
                           ignore_index, vec_ext)
        src = None
    else:
        _polygonize_raster(mask_path, save_path, None, None, ignore_index,
                           vec_ext)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask_path", type=str, required=True, \
                        help="Path of mask data.")
    parser.add_argument("--save_path", type=str, required=True, \
                        help="Path to save the shape file (the extension is .json/geojson or .shp).")
    parser.add_argument("--src_img_path", type=str, default="", \
                        help="Path of original data with geoinfo. Default to empty.")
    parser.add_argument("--ignore_index", type=int, default=255, \
                        help="The ignored index will not be converted to a value in the shape file. Default value is 255.")
    args = parser.parse_args()
    mask2shape(args.src_img_path, args.mask_path, args.save_path,
               args.ignore_index)
