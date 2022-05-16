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
import copy

import numpy as np
import shapely.ops
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection
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


# region flip
def img_flip(im, method=0):
    """
    flip image in different ways, this function provides 5 method to filp
    this function can be applied to 2D or 3D images

    Args:
        im(array): image array
        method(int or string): choose the flip method, it must be one of [
                                0, 1, 2, 3, 4, 'h', 'v', 'hv', 'rt2lb', 'lt2rb', 'dia', 'adia']
        0 or 'h': flipped in horizontal direction, which is the most frequently used method
        1 or 'v': flipped in vertical direction
        2 or 'hv': flipped in both horizontal diction and vertical direction
        3 or 'rt2lb' or 'dia': flipped around the diagonal,
                                which also can be thought as changing the RightTop part with LeftBottom part,
                                so it is called 'rt2lb' as well.
        4 or 'lt2rb' or 'adia': flipped around the anti-diagonal
                                    which also can be thought as changing the LeftTop part with RightBottom part,
                                    so it is called 'lt2rb' as well.

    Returns:
        flipped image(array)

    Raises:
        ValueError: Shape of image should 2d, 3d or more.

    Examples:
        --assume an image is like this:

        img:
        / + +
        - / *
        - * /

        --we can flip it in following code:

        img_h = im_flip(img, 'h')
        img_v = im_flip(img, 'v')
        img_vh = im_flip(img, 2)
        img_rt2lb = im_flip(img, 3)
        img_lt2rb = im_flip(img, 4)

        --we can get flipped image:

        img_h, flipped in horizontal direction
        + + \
        * \ -
        \ * -

        img_v, flipped in vertical direction
        - * \
        - \ *
        \ + +

        img_vh, flipped in both horizontal diction and vertical direction
        / * -
        * / -
        + + /

        img_rt2lb, flipped around the diagonal
        / | |
        + / *
        + * /

        img_lt2rb, flipped around the anti-diagonal
        / * +
        * / +
        | | /

    """
    if not len(im.shape) >= 2:
        raise ValueError("Shape of image should 2d, 3d or more")
    if method==0 or method=='h':
        return horizontal_flip(im)
    elif method==1 or method=='v':
        return vertical_flip(im)
    elif method==2 or method=='hv':
        return hv_flip(im)
    elif method==3 or method=='rt2lb' or method=='dia':
        return rt2lb_flip(im)
    elif method==4 or method=='lt2rb' or method=='adia':
        return lt2rb_flip(im)
    else:
        return im

def horizontal_flip(im):
    im = im[:, ::-1, ...]
    return im

def vertical_flip(im):
    im = im[::-1, :, ...]
    return im

def hv_flip(im):
    im = im[::-1, ::-1, ...]
    return im

def rt2lb_flip(im):
    axs_list = list(range(len(im.shape)))
    axs_list[:2] = [1, 0]
    im = im.transpose(axs_list)
    return im

def lt2rb_flip(im):
    axs_list = list(range(len(im.shape)))
    axs_list[:2] = [1, 0]
    im = im[::-1, ::-1, ...].transpose(axs_list)
    return im

# endregion

# region rotation
def img_simple_rotate(im, method=0):
    """
    rotate image in simple ways, this function provides 3 method to rotate
    this function can be applied to 2D or 3D images

    Args:
        im(array): image array
        method(int or string): choose the flip method, it must be one of [
                                0, 1, 2, 90, 180, 270
                                ]
        0 or 90 : rotated in 90 degree, clockwise
        1 or 180: rotated in 180 degree, clockwise
        2 or 270: rotated in 270 degree, clockwise

    Returns:
        flipped image(array)


    Raises:
        ValueError: Shape of image should 2d, 3d or more.


    Examples:
        --assume an image is like this:

        img:
        / + +
        - / *
        - * /

        --we can rotate it in following code:

        img_r90 = img_simple_rotate(img, 90)
        img_r180 = img_simple_rotate(img, 1)
        img_r270 = img_simple_rotate(img, 2)

        --we can get rotated image:

        img_r90, rotated in 90 degree
        | | \
        * \ +
        \ * +

        img_r180, rotated in 180 degree
        / * -
        * / -
        + + /

        img_r270, rotated in 270 degree
        + * \
        + \ *
        \ | |


    """
    if not len(im.shape) >= 2:
        raise ValueError("Shape of image should 2d, 3d or more")
    if method==0 or method==90:
        return rot_90(im)
    elif method==1 or method==180:
        return rot_180(im)
    elif method==2 or method==270:
        return rot_270(im)
    else:
        return im

def rot_90(im):
    axs_list = list(range(len(im.shape)))
    axs_list[:2] = [1, 0]
    im = im[::-1, :, ...].transpose(axs_list)
    return im

def rot_180(im):
    im = im[::-1, ::-1, ...]
    return im

def rot_270(im):
    axs_list = list(range(len(im.shape)))
    axs_list[:2] = [1, 0]
    im = im[:, ::-1, ...].transpose(axs_list)
    return im
# endregion


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


def to_uint8(im, is_linear=False):
    """ Convert raster to uint8.
    
    Args:
        im (np.ndarray): The image.
        is_linear (bool, optional): Use 2% linear stretch or not. Default is False.

    Returns:
        np.ndarray: Image on uint8.
    """

    # 2% linear stretch
    def _two_percent_linear(image, max_out=255, min_out=0):
        def _gray_process(gray, maxout=max_out, minout=min_out):
            # get the corresponding gray level at 98% histogram
            high_value = np.percentile(gray, 98)
            low_value = np.percentile(gray, 2)
            truncated_gray = np.clip(gray, a_min=low_value, a_max=high_value)
            processed_gray = ((truncated_gray - low_value) / (high_value - low_value)) * \
                             (maxout - minout)
            return np.uint8(processed_gray)

        if len(image.shape) == 3:
            processes = []
            for b in range(image.shape[-1]):
                processes.append(_gray_process(image[:, :, b]))
            result = np.stack(processes, axis=2)
        else:  # if len(image.shape) == 2
            result = _gray_process(image)
        return np.uint8(result)

    # simple image standardization
    def _sample_norm(image):
        stretches = []
        if len(image.shape) == 3:
            for b in range(image.shape[-1]):
                stretched = exposure.equalize_hist(image[:, :, b])
                stretched /= float(np.max(stretched))
                stretches.append(stretched)
            stretched_img = np.stack(stretches, axis=2)
        else:  # if len(image.shape) == 2
            stretched_img = exposure.equalize_hist(image)
        return np.uint8(stretched_img * 255)

    dtype = im.dtype.name
    dtypes = ["uint8", "uint16", "uint32", "float32"]
    if dtype not in dtypes:
        raise ValueError(
            f"'dtype' must be uint8/uint16/uint32/float32, not {dtype}.")
    if dtype != "uint8":
        im = _sample_norm(im)
    if is_linear:
        im = _two_percent_linear(im)
    return im


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
