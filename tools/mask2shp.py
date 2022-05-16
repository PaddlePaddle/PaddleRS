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

import numpy as np
from PIL import Image
try:
    from osgeo import gdal, ogr, osr
except ImportError:
    import gdal
    import ogr
    import osr

from utils import Raster, Timer


def _mask2tif(mask_path, tmp_path, proj, geot):
    mask = np.asarray(Image.open(mask_path))
    if len(mask.shape) == 3:
        mask = mask[:, :, 0]
    row, columns = mask.shape[:2]
    driver = gdal.GetDriverByName("GTiff")
    dst_ds = driver.Create(tmp_path, columns, row, 1, gdal.GDT_UInt16)
    dst_ds.SetGeoTransform(geot)
    dst_ds.SetProjection(proj)
    dst_ds.GetRasterBand(1).WriteArray(mask)
    dst_ds.FlushCache()
    return dst_ds


def _polygonize_raster(mask_path, shp_save_path, proj, geot, ignore_index):
    tmp_path = shp_save_path.replace(".shp", ".tif")
    ds = _mask2tif(mask_path, tmp_path, proj, geot)
    srcband = ds.GetRasterBand(1)
    maskband = srcband.GetMaskBand()
    gdal.SetConfigOption("GDAL_FILENAME_IS_UTF8", "YES")
    gdal.SetConfigOption("SHAPE_ENCODING", "UTF-8")
    ogr.RegisterAll()
    drv = ogr.GetDriverByName("ESRI Shapefile")
    if osp.exists(shp_save_path):
        os.remove(shp_save_path)
    dst_ds = drv.CreateDataSource(shp_save_path)
    prosrs = osr.SpatialReference(wkt=ds.GetProjection())
    dst_layer = dst_ds.CreateLayer(
        "Building boundary", geom_type=ogr.wkbPolygon, srs=prosrs)
    dst_fieldname = "DN"
    fd = ogr.FieldDefn(dst_fieldname, ogr.OFTInteger)
    dst_layer.CreateField(fd)
    gdal.Polygonize(srcband, maskband, dst_layer, 0, [])
    lyr = dst_ds.GetLayer()
    lyr.SetAttributeFilter("DN = '{}'".format(str(ignore_index)))
    for holes in lyr:
        lyr.DeleteFeature(holes.GetFID())
    dst_ds.Destroy()
    ds = None
    os.remove(tmp_path)


@Timer
def raster2shp(srcimg_path, mask_path, save_path, ignore_index=255):
    src = Raster(srcimg_path)
    _polygonize_raster(mask_path, save_path, src.proj, src.geot, ignore_index)
    src = None


parser = argparse.ArgumentParser(description="input parameters")
parser.add_argument("--srcimg_path", type=str, required=True, \
                    help="The path of original data with geoinfos.")
parser.add_argument("--mask_path", type=str, required=True, \
                    help="The path of mask data.")
parser.add_argument("--save_path", type=str, default="output", \
                    help="The path to save the results shapefile, `output` is the default.")
parser.add_argument("--ignore_index", type=int, default=255, \
                    help="It will not be converted to the value of SHP, `255` is the default.")

if __name__ == "__main__":
    args = parser.parse_args()
    raster2shp(args.srcimg_path, args.mask_path, args.save_path,
               args.ignore_index)
