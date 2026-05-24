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

# reference: https://zhuanlan.zhihu.com/p/378918221

try:
    from osgeo import gdal, ogr, osr
except:
    import gdal
    import ogr
    import osr


def translate_vector(geojson_path: str,
                     wo_wkt: str,
                     g_type: str="POLYGON",
                     dim: str="XY") -> str:
    ogr.RegisterAll()
    gdal.SetConfigOption("GDAL_FILENAME_IS_UTF8", "YES")
    data = ogr.Open(geojson_path)
    layer = data.GetLayer()
    spatial = layer.GetSpatialRef()
    layerName = layer.GetName()
    data.Destroy()
    dstSRS = osr.SpatialReference()
    dstSRS.ImportFromWkt(wo_wkt)
    ext = "." + geojson_path.split(".")[-1]
    save_path = geojson_path.replace(ext, ("_tmp" + ext))
    options = gdal.VectorTranslateOptions(
        srcSRS=spatial,
        dstSRS=dstSRS,
        reproject=True,
        layerName=layerName,
        geometryType=g_type,
        dim=dim)
    gdal.VectorTranslate(save_path, srcDS=geojson_path, options=options)
    return save_path
