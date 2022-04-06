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

import cv2
import numpy as np
import copy
import operator
import shapely.ops
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection
from functools import reduce
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from skimage import exposure


def normalize(im, mean, std, min_value=[0, 0, 0], max_value=[255, 255, 255]):
    # Rescaling (min-max normalization)
    range_value = np.asarray(
        [1. / (max_value[i] - min_value[i]) for i in range(len(max_value))],
        dtype=np.float32)
    im = (im - np.asarray(min_value, dtype=np.float32)) * range_value

    # Standardization (Z-score Normalization)
    im -= mean
    im /= std
    return im


def permute(im, to_bgr=False):
    im = np.swapaxes(im, 1, 2)
    im = np.swapaxes(im, 1, 0)
    if to_bgr:
        im = im[[2, 1, 0], :, :]
    return im


def center_crop(im, crop_size=224):
    height, width = im.shape[:2]
    w_start = (width - crop_size) // 2
    h_start = (height - crop_size) // 2
    w_end = w_start + crop_size
    h_end = h_start + crop_size
    im = im[h_start:h_end, w_start:w_end, ...]
    return im


def horizontal_flip(im):
    im = im[:, ::-1, ...]
    return im


def vertical_flip(im):
    im = im[::-1, :, ...]
    return im


def rgb2bgr(im):
    return im[:, :, ::-1]


def is_poly(poly):
    assert isinstance(poly, (list, dict)), \
        "Invalid poly type: {}".format(type(poly))
    return isinstance(poly, list)


def horizontal_flip_poly(poly, width):
    flipped_poly = np.array(poly)
    flipped_poly[0::2] = width - np.array(poly[0::2])
    return flipped_poly.tolist()


def horizontal_flip_rle(rle, height, width):
    import pycocotools.mask as mask_util
    if 'counts' in rle and type(rle['counts']) == list:
        rle = mask_util.frPyObjects(rle, height, width)
    mask = mask_util.decode(rle)
    mask = mask[:, ::-1]
    rle = mask_util.encode(np.array(mask, order='F', dtype=np.uint8))
    return rle


def vertical_flip_poly(poly, height):
    flipped_poly = np.array(poly)
    flipped_poly[1::2] = height - np.array(poly[1::2])
    return flipped_poly.tolist()


def vertical_flip_rle(rle, height, width):
    import pycocotools.mask as mask_util
    if 'counts' in rle and type(rle['counts']) == list:
        rle = mask_util.frPyObjects(rle, height, width)
    mask = mask_util.decode(rle)
    mask = mask[::-1, :]
    rle = mask_util.encode(np.array(mask, order='F', dtype=np.uint8))
    return rle


def crop_poly(segm, crop):
    xmin, ymin, xmax, ymax = crop
    crop_coord = [xmin, ymin, xmin, ymax, xmax, ymax, xmax, ymin]
    crop_p = np.array(crop_coord).reshape(4, 2)
    crop_p = Polygon(crop_p)

    crop_segm = list()
    for poly in segm:
        poly = np.array(poly).reshape(len(poly) // 2, 2)
        polygon = Polygon(poly)
        if not polygon.is_valid:
            exterior = polygon.exterior
            multi_lines = exterior.intersection(exterior)
            polygons = shapely.ops.polygonize(multi_lines)
            polygon = MultiPolygon(polygons)
        multi_polygon = list()
        if isinstance(polygon, MultiPolygon):
            multi_polygon = copy.deepcopy(polygon)
        else:
            multi_polygon.append(copy.deepcopy(polygon))
        for per_polygon in multi_polygon:
            inter = per_polygon.intersection(crop_p)
            if not inter:
                continue
            if isinstance(inter, (MultiPolygon, GeometryCollection)):
                for part in inter:
                    if not isinstance(part, Polygon):
                        continue
                    part = np.squeeze(
                        np.array(part.exterior.coords[:-1]).reshape(1, -1))
                    part[0::2] -= xmin
                    part[1::2] -= ymin
                    crop_segm.append(part.tolist())
            elif isinstance(inter, Polygon):
                crop_poly = np.squeeze(
                    np.array(inter.exterior.coords[:-1]).reshape(1, -1))
                crop_poly[0::2] -= xmin
                crop_poly[1::2] -= ymin
                crop_segm.append(crop_poly.tolist())
            else:
                continue
    return crop_segm


def crop_rle(rle, crop, height, width):
    import pycocotools.mask as mask_util
    if 'counts' in rle and type(rle['counts']) == list:
        rle = mask_util.frPyObjects(rle, height, width)
    mask = mask_util.decode(rle)
    mask = mask[crop[1]:crop[3], crop[0]:crop[2]]
    rle = mask_util.encode(np.array(mask, order='F', dtype=np.uint8))
    return rle


def expand_poly(poly, x, y):
    expanded_poly = np.array(poly)
    expanded_poly[0::2] += x
    expanded_poly[1::2] += y
    return expanded_poly.tolist()


def expand_rle(rle, x, y, height, width, h, w):
    import pycocotools.mask as mask_util
    if 'counts' in rle and type(rle['counts']) == list:
        rle = mask_util.frPyObjects(rle, height, width)
    mask = mask_util.decode(rle)
    expanded_mask = np.full((h, w), 0).astype(mask.dtype)
    expanded_mask[y:y + height, x:x + width] = mask
    rle = mask_util.encode(np.array(expanded_mask, order='F', dtype=np.uint8))
    return rle


def resize_poly(poly, im_scale_x, im_scale_y):
    resized_poly = np.array(poly, dtype=np.float32)
    resized_poly[0::2] *= im_scale_x
    resized_poly[1::2] *= im_scale_y
    return resized_poly.tolist()


def resize_rle(rle, im_h, im_w, im_scale_x, im_scale_y, interp):
    import pycocotools.mask as mask_util
    if 'counts' in rle and type(rle['counts']) == list:
        rle = mask_util.frPyObjects(rle, im_h, im_w)
    mask = mask_util.decode(rle)
    mask = cv2.resize(
        mask, None, None, fx=im_scale_x, fy=im_scale_y, interpolation=interp)
    rle = mask_util.encode(np.array(mask, order='F', dtype=np.uint8))
    return rle


def to_uint8(im):
    """ Convert raster to uint8.
    
    Args:
        im (np.ndarray): The image.

    Returns:
        np.ndarray: Image on uint8.
    """

    # 2% linear stretch
    def _two_percentLinear(image, max_out=255, min_out=0):
        def _gray_process(gray, maxout=max_out, minout=min_out):
            # get the corresponding gray level at 98% histogram
            high_value = np.percentile(gray, 98)
            low_value = np.percentile(gray, 2)
            truncated_gray = np.clip(gray, a_min=low_value, a_max=high_value)
            processed_gray = ((truncated_gray - low_value) / (high_value - low_value)) * \
                             (maxout - minout)
            return processed_gray

        if len(image.shape) == 3:
            processes = []
            for b in range(image.shape[-1]):
                processes.append(_gray_process(image[:, :, b]))
            result = np.stack(processes, axis=2)
        else:  # if len(image.shape) == 2
            result = _gray_process(image)
        return np.uint8(result)

    # simple image standardization
    def _sample_norm(image, NUMS=65536):
        stretches = []
        if len(image.shape) == 3:
            for b in range(image.shape[-1]):
                stretched = _stretch(image[:, :, b], NUMS)
                stretched /= float(NUMS)
                stretches.append(stretched)
            stretched_img = np.stack(stretches, axis=2)
        else:  # if len(image.shape) == 2
            stretched_img = _stretch(image, NUMS)
        return np.uint8(stretched_img * 255)

    # histogram equalization
    def _stretch(ima, NUMS):
        hist = _histogram(ima, NUMS)
        lut = []
        for bt in range(0, len(hist), NUMS):
            # step size
            step = reduce(operator.add, hist[bt:bt + NUMS]) / (NUMS - 1)
            # create balanced lookup table
            n = 0
            for i in range(NUMS):
                lut.append(n / step)
                n += hist[i + bt]
            np.take(lut, ima, out=ima)
            return ima

    # calculate histogram
    def _histogram(ima, NUMS):
        bins = list(range(0, NUMS))
        flat = ima.flat
        n = np.searchsorted(np.sort(flat), bins)
        n = np.concatenate([n, [len(flat)]])
        hist = n[1:] - n[:-1]
        return hist

    dtype = im.dtype.name
    dtypes = ["uint8", "uint16", "float32"]
    if dtype not in dtypes:
        raise ValueError(f"'dtype' must be uint8/uint16/float32, not {dtype}.")
    if dtype == "uint8":
        return im
    else:
        if dtype == "float32":
            im = _sample_norm(im)
        return _two_percentLinear(im)


def to_intensity(im):
    """ calculate SAR data's intensity diagram.

    Args:
        im (np.ndarray): The SAR image.

    Returns:
        np.ndarray: Intensity diagram.
    """
    if len(im.shape) != 2:
        raise ValueError("im's shape must be 2.")
    # the type is complex means this is a SAR data
    if isinstance(type(im[0, 0]), complex):
        im = abs(im)
    return im


def select_bands(im, band_list=[1, 2, 3]):
    """ Select bands.

    Args:
        im (np.ndarray): The image.
        band_list (list, optional): Bands of selected (Start with 1). Defaults to [1, 2, 3].

    Returns:
        np.ndarray: The image after band selected.
    """
    if len(im.shape) == 2:  # just have one channel
        return im
    if not isinstance(band_list, list) or len(band_list) == 0:
        raise TypeError("band_list must be non empty list.")
    total_band = im.shape[-1]
    result = []
    for band in band_list:
        band = int(band - 1)
        if band < 0 or band >= total_band:
            raise ValueError("The element in band_list must > 1 and <= {}.".
                             format(str(total_band)))
        result.append(im[:, :, band])
    ima = np.stack(result, axis=-1)
    return ima


def de_haze(im, gamma=False):
    """ Priori defogging of dark channel. (Just RGB)

    Args:
        im (np.ndarray): The image.
        gamma (bool, optional): Use gamma correction or not. Defaults to False.

    Returns:
        np.ndarray: The image after defogged.
    """

    def _guided_filter(I, p, r, eps):
        m_I = cv2.boxFilter(I, -1, (r, r))
        m_p = cv2.boxFilter(p, -1, (r, r))
        m_Ip = cv2.boxFilter(I * p, -1, (r, r))
        cov_Ip = m_Ip - m_I * m_p
        m_II = cv2.boxFilter(I * I, -1, (r, r))
        var_I = m_II - m_I * m_I
        a = cov_Ip / (var_I + eps)
        b = m_p - a * m_I
        m_a = cv2.boxFilter(a, -1, (r, r))
        m_b = cv2.boxFilter(b, -1, (r, r))
        return m_a * I + m_b

    def _de_fog(im, r, w, maxatmo_mask, eps):
        # im is RGB and range[0, 1]
        atmo_mask = np.min(im, 2)
        dark_channel = cv2.erode(atmo_mask, np.ones((15, 15)))
        atmo_mask = _guided_filter(atmo_mask, dark_channel, r, eps)
        bins = 2000
        ht = np.histogram(atmo_mask, bins)
        d = np.cumsum(ht[0]) / float(atmo_mask.size)
        for lmax in range(bins - 1, 0, -1):
            if d[lmax] <= 0.999:
                break
        atmo_illum = np.mean(im, 2)[atmo_mask >= ht[1][lmax]].max()
        atmo_mask = np.minimum(atmo_mask * w, maxatmo_mask)
        return atmo_mask, atmo_illum

    if np.max(im) > 1:
        im = im / 255.
    result = np.zeros(im.shape)
    mask_img, atmo_illum = _de_fog(
        im, r=81, w=0.95, maxatmo_mask=0.80, eps=1e-8)
    for k in range(3):
        result[:, :, k] = (im[:, :, k] - mask_img) / (1 - mask_img / atmo_illum)
    result = np.clip(result, 0, 1)
    if gamma:
        result = result**(np.log(0.5) / np.log(result.mean()))
    return (result * 255).astype("uint8")


def pca(im, dim=3, whiten=True):
    """ Dimensionality reduction of PCA. 

    Args:
        im (np.ndarray): The image.
        dim (int, optional): Reserved dimensions. Defaults to 3.
        whiten (bool, optional): PCA whiten or not. Defaults to True.

    Returns:
        np.ndarray: The image after PCA.
    """
    H, W, C = im.shape
    n_im = np.reshape(im, (-1, C))
    pca = PCA(n_components=dim, whiten=whiten)
    im_pca = pca.fit_transform(n_im)
    result = np.reshape(im_pca, (H, W, dim))
    result = np.clip(result, 0, 1)
    return (result * 255).astype("uint8")


def match_histograms(im, ref):
    """
    Match the cumulative histogram of one image to another.

    Args:
        im (np.ndarray): The input image.
        ref (np.ndarray): The reference image to match histogram of. `ref` must have the same number of channels as `im`.

    Returns:
        np.ndarray: The transformed input image.

    Raises:
        ValueError: When the number of channels of `ref` differs from that of im`.
    """
    # TODO: Check the data types of the inputs to see if they are supported by skimage
    return exposure.match_histograms(
        im, ref, channel_axis=-1 if im.ndim > 2 else None)


def match_by_regression(im, ref, pif_loc=None):
    """
    Match the brightness values of two images using a linear regression method.

    Args:
        im (np.ndarray): The input image.
        ref (np.ndarray): The reference image to match. `ref` must have the same shape as `im`.
        pif_loc (tuple|None, optional): The spatial locations where pseudo-invariant features (PIFs) are obtained. If 
            `pif_loc` is set to None, all pixels in the image will be used as training samples for the regression model. 
            In other cases, `pif_loc` should be a tuple of np.ndarrays. Default: None.

    Returns:
        np.ndarray: The transformed input image.

    Raises:
        ValueError: When the shape of `ref` differs from that of `im`.
    """

    def _linear_regress(im, ref, loc):
        regressor = LinearRegression()
        if loc is not None:
            x, y = im[loc], ref[loc]
        else:
            x, y = im, ref
        x, y = x.reshape(-1, 1), y.ravel()
        regressor.fit(x, y)
        matched = regressor.predict(im.reshape(-1, 1))
        return matched.reshape(im.shape)

    if im.shape != ref.shape:
        raise ValueError("Image and Reference must have the same shape!")

    if im.ndim > 2:
        # Multiple channels
        matched = np.empty(im.shape, dtype=im.dtype)
        for ch in range(im.shape[-1]):
            matched[..., ch] = _linear_regress(im[..., ch], ref[..., ch],
                                               pif_loc)
    else:
        # Single channel
        matched = _linear_regress(im, ref, pif_loc).astype(im.dtype)

    return matched
