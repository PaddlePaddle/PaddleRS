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

import os.path as osp
from typing import List, Tuple, Union, Optional
import webbrowser
import numpy as np
from folium import folium, Map, LayerControl
from folium.raster_layers import ImageOverlay
from paddlers.transforms.functions import to_uint8

try:
    from osgeo import gdal, osr
except:
    import gdal
    import osr


def map_show(
        raster_path: str,
        band_list: Union[List[int], Tuple[int, ...], None]=None,
        save_path: Optional[str]=None, ) -> folium.Map:
    """Show raster in folium online map.

    Args:
        raster_path (str): The path of raster.
        band_list (Union[List[int], Tuple[int, ...], None], optional): 
            Select a set of bands (the band index starts from 1). 
            If None, read all bands. Defaults to None.
        save_path (str, optional): The path of save html. 
            If None, it is only display on Jupyter Notebook.
            Defaults to None.

    Returns:
        folium.Map: An example of folium map.
    """
    fmap = Map(max_zoom=24, prefer_canvas=True)
    layer, center = Raster(raster_path, band_list).get_layer()
    layer.add_to(fmap)
    if center is not None:
        fmap.location = center
    fmap.fit_bounds(layer.bounds)
    LayerControl().add_to(fmap)
    if save_path:
        fmap.save(save_path)
        webbrowser.open(save_path)
    return fmap


class Raster:
    def __init__(
            self,
            path: str,
            band_list: Union[List[int], Tuple[int, ...], None]=None) -> None:
        self.src_data = Converter.open_as_WGS84(path)
        if self.src_data is None:
            raise ValueError("Faild to open {}.".format(path))
        self.name = path.replace("\\", "/").split("/")[-1].split(".")[0]
        self.set_bands(band_list)
        self._get_info()

    def get_layer(self) -> ImageOverlay:
        layer = ImageOverlay(self._get_array(), self.wgs_range, name=self.name)
        return layer, self.wgs_center

    def set_bands(self,
                  band_list: Union[List[int], Tuple[int, ...], None]) -> None:
        self.bands = self.src_data.RasterCount
        if band_list is None:
            if self.bands == 3:
                band_list = [1, 2, 3]
            else:
                band_list = [1]
        band_list_lens = len(band_list)
        if band_list_lens not in (1, 3):
            raise ValueError("The lenght of band_list must be 1 or 3, not {}.".
                             format(str(band_list_lens)))
        if max(band_list) > self.bands or min(band_list) < 1:
            raise ValueError("The range of band_list must within [1, {}].".
                             format(str(self.bands)))
        self.band_list = band_list

    def _get_info(self) -> None:
        self.width = self.src_data.RasterXSize
        self.height = self.src_data.RasterYSize
        self.geotf = self.src_data.GetGeoTransform()
        self.proj = self.src_data.GetProjection()  # WGS84
        self.wgs_range = self._get_WGS84_range()
        self.wgs_center = self._get_WGS84_center()

    def _get_WGS84_range(self) -> List:
        converter = Converter(self.proj, self.geotf)
        lat1, lon1 = converter.xy2latlon(self.height - 1, 0)
        lat2, lon2 = converter.xy2latlon(0, self.width - 1)
        return [[lon1, lat1], [lon2, lat2]]

    def _get_WGS84_center(self) -> List:
        clat = (self.wgs_range[0][0] + self.wgs_range[1][0]) / 2
        clon = (self.wgs_range[0][1] + self.wgs_range[1][1]) / 2
        return [clat, clon]

    def _get_array(self) -> np.ndarray:
        band_array = []
        for b in self.band_list:
            band_i = self.src_data.GetRasterBand(b).ReadAsArray()
            band_array.append(band_i)
        ima = np.stack(band_array, axis=0)
        if self.bands == 1:
            # the type is complex means this is a SAR data
            if isinstance(type(ima[0, 0]), complex):
                ima = abs(ima)
            ima = ima.squeeze()
        else:
            ima = ima.transpose((1, 2, 0))
        ima = to_uint8(ima, True)
        return ima


class Converter:
    def __init__(self, proj: str, geotf: tuple) -> None:
        # source data
        self.source = osr.SpatialReference()
        self.source.ImportFromWkt(proj)
        self.geotf = geotf
        # target data
        self.target = osr.SpatialReference()
        self.target.ImportFromEPSG(4326)

    @classmethod
    def open_as_WGS84(self, path: str) -> gdal.Dataset:
        if not osp.exists(path):
            raise FileNotFoundError("{} not found.".format(path))
        result = gdal.Warp("", path, dstSRS="EPSG:4326", format="VRT")
        return result

    def xy2latlon(self, row: int, col: int) -> List:
        px = self.geotf[0] + col * self.geotf[1] + row * self.geotf[2]
        py = self.geotf[3] + col * self.geotf[4] + row * self.geotf[5]
        ct = osr.CoordinateTransformation(self.source, self.target)
        coords = ct.TransformPoint(px, py)
        return coords[:2]