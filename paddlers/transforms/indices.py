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

# Refer to https://github.com/awesome-spectral-indices/awesome-spectral-indices .
# See LICENSE (https://github.com/awesome-spectral-indices/awesome-spectral-indices/blob/main/LICENSE).

import abc

__all__ = [
    'ARI', 'ARI2', 'ARVI', 'AWEInsh', 'AWEIsh', 'BAI', 'BI', 'BLFEI', 'BNDVI',
    'BWDRVI', 'BaI', 'CIG', 'CSI', 'CSIT', 'DBI', 'DBSI', 'DVI', 'EBBI', 'EMBI',
    'EVI', 'EVI2', 'FCVI', 'GARI', 'GBNDVI', 'GLI', 'GNDVI', 'GRVI', 'IPVI',
    'LSWI', 'MBI', 'MGRVI', 'MNDVI', 'MNDWI', 'MNLI', 'MSI', 'NBLI', 'NDSI',
    'NDVI', 'NDWI', 'NDYI', 'NIRv', 'PSRI', 'RI', 'SAVI', 'SWI', 'TDVI', 'UI',
    'VIG', 'WI1', 'WI2', 'WRI'
]

EPS = 1e-32
BAND_NAMES = [
    "b", "g", "r", "re1", "re2", "re3", "n", "s1", "s2", "t", "t1", "t2"
]

# | Band name |   Description    | Wavelength (μm) | Satellite |
# |-----------|------------------|-----------------|-----------|
# |     b     | Blue             |   0.450-0.515   | Landsat8  |
# |     g     | Green            |   0.525-0.600   | Landsat8  |
# |     r     | Red              |   0.630-0.680   | Landsat8  |
# |    re1    | Red Edge 1       |   0.698-0.713   | Sentinel2 |
# |    re2    | Red Edge 2       |   0.733-0.748   | Sentinel2 |
# |    re3    | Red Edge 3       |   0.773-0.793   | Sentinel2 |
# |     n     | NIR              |   0.845-0.885   | Landsat8  |
# |    s1     | SWIR 1           |   1.560-1.660   | Landsat8  |
# |    s2     | SWIR 2           |   2.100-2.300   | Landsat8  |
# |     t     | Thermal Infrared |   10.40-12.50   | Landsat7  |
# |    t1     | Thermal 1        |   10.60-11.19   | Landsat8  |
# |    t2     | Thermal 2        |   11.50-12.51   | Landsat8  |


class RSIndex(metaclass=abc.ABCMeta):
    def __init__(self, band_indices):
        super(RSIndex, self).__init__()
        self.band_indices = band_indices
        self.required_band_names = iintersection(
            self._compute.__code__.co_varnames[1:],  # strip self 
            BAND_NAMES  # only save band names
        )

    @abc.abstractmethod
    def _compute(self, *args, **kwargs):
        pass

    def __call__(self, image):
        bands = self.select_bands(image)
        now_band_names = tuple(bands.keys())
        if not iequal(now_band_names, self.required_band_names):
            raise LackBandError("Lack of bands: {}.".format(
                isubtraction(self.required_band_names, now_band_names)))
        return self._compute(**bands)

    def select_bands(self, image, to_float32=True):
        bands = {}
        for name, idx in self.band_indices.items():
            if name in self.required_band_names:
                if idx == 0:
                    raise ValueError("Band index starts from 1.")
                bands[name] = image[..., idx - 1]
                if to_float32:
                    bands[name] = bands[name].astype('float32')
        return bands


class LackBandError(Exception):
    pass


def iintersection(iter1, iter2):
    return tuple(set(iter1) & set(iter2))


def isubtraction(iter1, iter2):
    return tuple(set(iter1) - set(iter2))


def iequal(iter1, iter2):
    return set(iter1) == set(iter2)


def compute_normalized_difference_index(band1, band2):
    return (band1 - band2) / (band1 + band2 + EPS)


class ARI(RSIndex):
    def _compute(self, g, re1):
        index = 1 / (g + EPS)
        index -= 1 / (re1 + EPS)
        return index


class ARI2(RSIndex):
    def _compute(self, g, re1, n):
        index = 1 / (g + EPS)
        index -= 1 / (re1 + EPS)
        index = index * n
        return index


class ARVI(RSIndex):
    def __init__(self, band_indices, c0):
        super(ARVI, self).__init__(band_indices)
        self.c0 = c0

    def _compute(self, b, r, n):
        return compute_normalized_difference_index(n, r - self.c0 * (r - b))


class AWEInsh(RSIndex):
    def _compute(self, g, n, s1, s2):
        index = 4.0 * (g - s1)
        index -= 0.25 * n
        index += 2.75 * s2
        return index


class AWEIsh(RSIndex):
    def _compute(self, b, g, n, s1, s2):
        index = 2.5 * g
        index += b
        index -= 1.5 * (n + s1)
        index -= 0.25 * s2
        return index


class BAI(RSIndex):
    def _compute(self, r, n):
        index = (0.1 - r)**2.0
        index += (0.06 - n)**2.0
        return 1.0 / (index + EPS)


class BI(RSIndex):
    def _compute(self, b, r, n, s1):
        return compute_normalized_difference_index(s1 + r, n + b)


class BLFEI(RSIndex):
    def _compute(self, g, r, s1, s2):
        return compute_normalized_difference_index((g + r + s2) / 3.0, s1)


class BNDVI(RSIndex):
    def _compute(self, b, n):
        return compute_normalized_difference_index(n, b)


class BWDRVI(RSIndex):
    def __init__(self, band_indices, c0):
        super(BWDRVI, self).__init__(band_indices)
        self.c0 = c0

    def _compute(self, b, n):
        return compute_normalized_difference_index(self.c0 * n, b)


class BaI(RSIndex):
    def _compute(self, r, n, s1):
        index = r + s1
        index -= n
        return index


class CIG(RSIndex):
    def _compute(self, g, n):
        index = n / (g + EPS)
        index -= 1.0
        return index


class CSI(RSIndex):
    def _compute(self, n, s2):
        return n / (s2 + EPS)


class CSIT(RSIndex):
    def _compute(self, n, s2, t):
        return n / ((s2 * t) / 10000.0 + EPS)


class DBI(RSIndex):
    def _compute(self, b, r, n, t1):
        index = (b - t1) / (b + t1 + EPS)
        index -= (n - r) / (n + r + EPS)
        return index


class DBSI(RSIndex):
    def _compute(self, g, r, n, s1):
        index = (s1 - g) / (s1 + g + EPS)
        index -= (n - r) / (n + r + EPS)
        return index


class DVI(RSIndex):
    def _compute(self, r, n):
        return n - r


class EBBI(RSIndex):
    def _compute(self, n, s1, t):
        num = s1 - n
        denom = (10.0 * ((s1 + t)**0.5))
        return num / (denom + EPS)


class EMBI(RSIndex):
    def _compute(self, g, n, s1, s2):
        item1 = compute_normalized_difference_index(s1, s2 + n)
        item1 += 0.5
        item2 = compute_normalized_difference_index(g, s1)
        return (item1 - item2 - 0.5) / (item1 + item2 + 1.5 + EPS)


class EVI(RSIndex):
    def __init__(self, band_indices, c0=2.5, c1=6, c2=7.5, c3=1):
        super(EVI, self).__init__(band_indices)
        self.c0 = c0
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3

    def _compute(self, b, r, n):
        num = self.c0 * (n - r)
        denom = n + self.c1 * r - self.c2 * b + self.c3
        return num / (denom + EPS)


class EVI2(RSIndex):
    def __init__(self, band_indices, c0, c1):
        super(EVI2, self).__init__(band_indices)
        self.c0 = c0
        self.c1 = c1

    def _compute(self, n, r):
        num = self.c0 * (n - r)
        denom = n + 2.4 * r + self.c1
        return num / (denom + EPS)


class FCVI(RSIndex):
    def _compute(self, b, g, r, n):
        return n - ((r + g + b) / 3.0)


class GARI(RSIndex):
    def _compute(self, b, g, r, n):
        num = n - (g - (b - r))
        denom = n - (g + (b - r))
        return num / (denom + EPS)


class GBNDVI(RSIndex):
    def _compute(self, b, g, n):
        return compute_normalized_difference_index(n, g + b)


class GLI(RSIndex):
    def _compute(self, b, g, r):
        return compute_normalized_difference_index(2.0 * g, r + b)


class GNDVI(RSIndex):
    def _compute(self, g, n):
        return compute_normalized_difference_index(n, g)


class GRVI(RSIndex):
    def _compute(self, g, n):
        return n / (g + EPS)


class IPVI(RSIndex):
    def _compute(self, r, n):
        return n / (n + r + EPS)


class LSWI(RSIndex):
    def _compute(self, n, s1):
        return compute_normalized_difference_index(n, s1)


class MBI(RSIndex):
    def _compute(self, n, s1, s2):
        index = compute_normalized_difference_index(s1, s2 + n)
        index += 0.5
        return index


class MGRVI(RSIndex):
    def _compute(self, g, r):
        return compute_normalized_difference_index(g**2.0, r**2.0)


class MNDVI(RSIndex):
    def _compute(self, n, s2):
        return compute_normalized_difference_index(n, s2)


class MNDWI(RSIndex):
    def _compute(self, g, s1):
        return compute_normalized_difference_index(g, s1)


class MNLI(RSIndex):
    def __init__(self, band_indices, c0):
        super(MNLI, self).__init__(band_indices)
        self.c0 = c0

    def _compute(self, r, n):
        num = (1 + self.c0) * ((n**2) - r)
        denom = ((n**2) + r + self.c0)
        return num / (denom + EPS)


class MSI(RSIndex):
    def _compute(self, n, s1):
        return s1 / (n + EPS)


class NBLI(RSIndex):
    def _compute(self, r, t):
        return compute_normalized_difference_index(r, t)


class NDSI(RSIndex):
    def _compute(self, g, s1):
        return compute_normalized_difference_index(g, s1)


class NDVI(RSIndex):
    def _compute(self, r, n):
        return compute_normalized_difference_index(n, r)


class NDWI(RSIndex):
    def _compute(self, g, n):
        return compute_normalized_difference_index(g, n)


class NDYI(RSIndex):
    def _compute(self, b, g):
        return compute_normalized_difference_index(g, b)


class NIRv(RSIndex):
    def _compute(self, r, n):
        return compute_normalized_difference_index(n, r) * n


class PSRI(RSIndex):
    def _compute(self, b, r, re2):
        return (r - b) / (re2 + EPS)


class RI(RSIndex):
    def _compute(self, g, r):
        return compute_normalized_difference_index(r, g)


class SAVI(RSIndex):
    def __init__(self, band_indices, c0):
        super(SAVI, self).__init__(band_indices)
        self.c0 = c0

    def _compute(self, r, n):
        num = (1.0 + self.c0) * (n - r)
        denom = n + r + self.c0
        return num / (denom + EPS)


class SWI(RSIndex):
    def _compute(self, g, n, s1):
        num = g * (n - s1)
        denom = (g + n) * (n + s1)
        return num / (denom + EPS)


class TDVI(RSIndex):
    def _compute(self, r, n):
        num = 1.5 * (n - r)
        denom = (n**2.0 + r + 0.5)**0.5
        return num / (denom + EPS)


class UI(RSIndex):
    def _compute(self, n, s2):
        return compute_normalized_difference_index(s2, n)


class VIG(RSIndex):
    def _compute(self, g, r):
        return compute_normalized_difference_index(g, r)


class WI1(RSIndex):
    def _compute(self, g, s2):
        return compute_normalized_difference_index(g, s2)


class WI2(RSIndex):
    def _compute(self, b, s2):
        return compute_normalized_difference_index(b, s2)


class WRI(RSIndex):
    def _compute(self, g, r, n, s1):
        return (g + r) / (n + s1 + EPS)
