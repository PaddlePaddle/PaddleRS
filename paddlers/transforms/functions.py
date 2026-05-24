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

import copy

import cv2
import numpy as np
import shapely.ops
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection
from sklearn.linear_model import LinearRegression
from skimage import exposure
from joblib import load
from PIL import Image


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


def permute(im):
    im = np.swapaxes(im, 1, 2)
    im = np.swapaxes(im, 1, 0)
    return im


def center_crop(im, crop_size=224):
    height, width = im.shape[:2]
    w_start = (width - crop_size) // 2
    h_start = (height - crop_size) // 2
    w_end = w_start + crop_size
    h_end = h_start + crop_size
    im = im[h_start:h_end, w_start:w_end, ...]
    return im


def img_flip(im, method=0):
    """
    Flip an image. 
    This function provides 5 flipping methods and can be applied to 2D or 3D numpy arrays.

    Args:
        im (np.ndarray): Input image.
        method (int|string): Flipping method. Must be one of [
                                0, 1, 2, 3, 4, 'h', 'v', 'hv', 'rt2lb', 'lt2rb', 
                                'dia', 'adia'].
            0 or 'h': flip the image in horizontal direction, which is the most frequently 
                used method;
            1 or 'v': flip the image in vertical direction;
            2 or 'hv': flip the image in both horizontal diction and vertical direction;
            3 or 'rt2lb' or 'dia': flip the image across the diagonal;
            4 or 'lt2rb' or 'adia': flip the image across the anti-diagonal.

    Returns:
        np.ndarray: Flipped image.

    Raises:
        ValueError: Invalid shape of images.

    Examples:
        Assume an image is like this:

        img:
        / + +
        - / *
        - * /

        We can flip it with following code:

        img_h = img_flip(img, 'h')
        img_v = img_flip(img, 'v')
        img_vh = img_flip(img, 2)
        img_rt2lb = img_flip(img, 3)
        img_lt2rb = img_flip(img, 4)

        Then we get the flipped images:

        img_h, flipped in horizontal direction:
        + + \
        * \ -
        \ * -

        img_v, flipped in vertical direction:
        - * \
        - \ *
        \ + +

        img_vh, flipped in both horizontal diction and vertical direction:
        / * -
        * / -
        + + /

        img_rt2lb, mirrored on the diagonal:
        / | |
        + / *
        + * /

        img_lt2rb, mirrored on the anti-diagonal:
        / * +
        * / +
        | | /
    """

    if not len(im.shape) >= 2:
        raise ValueError("The number of image dimensions is less than 2.")
    if method == 0 or method == 'h':
        return horizontal_flip(im)
    elif method == 1 or method == 'v':
        return vertical_flip(im)
    elif method == 2 or method == 'hv':
        return hv_flip(im)
    elif method == 3 or method == 'rt2lb' or method == 'dia':
        return rt2lb_flip(im)
    elif method == 4 or method == 'lt2rb' or method == 'adia':
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


def img_simple_rotate(im, method=0):
    """
    Rotate an image. 
    This function provides 3 rotating methods and can be applied to 2D or 3D numpy arrays.

    Args:
        im (np.ndarray): Input image.
        method (int|string): Rotating method, which must be one of [
                                0, 1, 2, 90, 180, 270
                                ].
            0 or 90 : rotate the image by 90 degrees, clockwise;
            1 or 180: rotate the image by 180 degrees, clockwise;
            2 or 270: rotate the image by 270 degrees, clockwise.

    Returns:
        np.ndarray: Rotated image.

    Raises:
        ValueError: Invalid shape of images.

    Examples:
        Assume an image is like this:

        img:
        / + +
        - / *
        - * /

        We can rotate it with following code:

        img_r90 = img_simple_rotate(img, 90)
        img_r180 = img_simple_rotate(img, 1)
        img_r270 = img_simple_rotate(img, 2)

        Then we get the following rotated images:

        img_r90, rotated by 90°:
        | | \
        * \ +
        \ * +

        img_r180, rotated by 180°:
        / * -
        * / -
        + + /

        img_r270, rotated by 270°:
        + * \
        + \ *
        \ | |
    """

    if not len(im.shape) >= 2:
        raise ValueError("The number of image dimensions is less than 2.")
    if method == 0 or method == 90:
        return rot_90(im)
    elif method == 1 or method == 180:
        return rot_180(im)
    elif method == 2 or method == 270:
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


def to_uint8(im, norm=True, stretch=False):
    """
    Convert raster data to uint8 type.
    
    Args:
        im (np.ndarray): Input raster image.
        norm (bool, optional): Use hist equalization to normalize each band or not. 
            Default is True.
        stretch (bool, optional): Use 2% linear stretch or not. Default is False.

    Returns:
        np.ndarray: Image data with unit8 type.
    """

    EPS = 1e-32

    def _minmax_norm(image):
        image = image.astype(np.float32)
        min_val = image.min()
        max_val = image.max()
        return (image - min_val) / (max_val - min_val + EPS)

    # 2% linear stretch
    def _two_percent_linear(image, max_out=1., min_out=0.):
        def _gray_process(gray, maxout=max_out, minout=min_out):
            # Get the corresponding gray level at 98% in the histogram.
            high_value = np.percentile(gray, 98)
            low_value = np.percentile(gray, 2)
            truncated_gray = np.clip(gray, a_min=low_value, a_max=high_value)
            processed_gray = ((truncated_gray - low_value) / (high_value - low_value + EPS)) * \
                             (maxout - minout)
            return processed_gray

        if len(image.shape) == 3:
            processes = []
            for b in range(image.shape[-1]):
                processes.append(_gray_process(image[:, :, b]))
            result = np.stack(processes, axis=2)
        else:  # if len(image.shape) == 2
            result = _gray_process(image)
        return result

    def _equalize_hist(image):
        stretches = []
        if len(image.shape) == 3:
            for b in range(image.shape[-1]):
                stretched = exposure.equalize_hist(image[:, :, b])
                assert np.min(stretched) >= 0
                stretched /= float(np.max(stretched)) + EPS
                stretches.append(stretched)
            stretched_img = np.stack(stretches, axis=2)
        else:  # if len(image.shape) == 2
            stretched_img = exposure.equalize_hist(image)
            assert np.min(stretched_img) >= 0
            stretched_img /= float(np.max(stretched_img)) + EPS
        return stretched_img

    dtype = im.dtype.name
    if dtype == 'uint8':
        return im
    if stretch:
        im = _two_percent_linear(im)
    if norm:
        im = _equalize_hist(im)
    if not norm and not stretch:
        im = _minmax_norm(im)
    im = np.uint8(im * 255)
    return im


def to_intensity(im):
    """
    Calculate the intensity of SAR data.

    Args:
        im (np.ndarray): SAR image.

    Returns:
        np.ndarray: Intensity image.
    """

    if len(im.shape) != 2:
        raise ValueError("`len(im.shape) must be 2.")
    # If the type is complex, this is SAR data.
    if isinstance(type(im[0, 0]), complex):
        im = abs(im)
    return im


def select_bands(im, band_list=[1, 2, 3]):
    """
    Select bands of a multi-band image.

    Args:
        im (np.ndarray): Input image.
        band_list (list, optional): Bands to select (band index start from 1). 
            Defaults to [1, 2, 3].

    Returns:
        np.ndarray: Image with selected bands.
    """

    if len(im.shape) == 2:  # Image has only one channel
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


def dehaze(im, gamma=False):
    """
    Perform single image haze removal using dark channel prior.

    Args:
        im (np.ndarray): Input image.
        gamma (bool, optional): Use gamma correction or not. Defaults to False.

    Returns:
        np.ndarray: Output dehazed image.
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

    def _dehaze(im, r, w, maxatmo_mask, eps):
        # im is a RGB image and the value ranges in [0, 1].
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
    mask_img, atmo_illum = _dehaze(
        im, r=81, w=0.95, maxatmo_mask=0.80, eps=1e-8)
    for k in range(3):
        result[:, :, k] = (im[:, :, k] - mask_img) / (1 - mask_img / atmo_illum)
    result = np.clip(result, 0, 1)
    if gamma:
        result = result**(np.log(0.5) / np.log(result.mean()))
    return (result * 255).astype("uint8")


def match_histograms(im, ref):
    """
    Match the cumulative histogram of one image to another.

    Args:
        im (np.ndarray): Input image.
        ref (np.ndarray): Reference image to match histogram of. `ref` must have 
            the same number of channels as `im`.

    Returns:
        np.ndarray: Transformed input image.

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
        im (np.ndarray): Input image.
        ref (np.ndarray): Reference image to match. `ref` must have the same shape 
            as `im`.
        pif_loc (tuple|None, optional): Spatial locations where pseudo-invariant 
            features (PIFs) are obtained. If `pif_loc` is set to None, all pixels in 
            the image will be used as training samples for the regression model. In 
            other cases, `pif_loc` should be a tuple of np.ndarrays. Default: None.

    Returns:
        np.ndarray: Transformed input image.

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


def match_lf_components(im, ref, lf_ratio=0.01):
    """
    Match the low-frequency components of two images.

    Args:
        im (np.ndarray): Input image.
        ref (np.ndarray): Reference image to match. `ref` must have the same shape 
            as `im`.
        lf_ratio (float, optional): Proportion of frequence components that should
            be recognized as low-frequency components in the frequency domain. 
            Default: 0.01.

    Returns:
        np.ndarray: Transformed input image.

    Raises:
        ValueError: When the shape of `ref` differs from that of `im`.
    """

    def _replace_lf(im, ref, lf_ratio):
        h, w = im.shape
        h_lf, w_lf = int(h // 2 * lf_ratio), int(w // 2 * lf_ratio)
        freq_im = np.fft.fft2(im)
        freq_ref = np.fft.fft2(ref)
        if h_lf > 0:
            freq_im[:h_lf] = freq_ref[:h_lf]
            freq_im[-h_lf:] = freq_ref[-h_lf:]
        if w_lf > 0:
            freq_im[:, :w_lf] = freq_ref[:, :w_lf]
            freq_im[:, -w_lf:] = freq_ref[:, -w_lf:]
        recon_im = np.fft.ifft2(freq_im)
        recon_im = np.abs(recon_im)
        return recon_im

    if im.shape != ref.shape:
        raise ValueError("Image and Reference must have the same shape!")

    if im.ndim > 2:
        # Multiple channels
        matched = np.empty(im.shape, dtype=im.dtype)
        for ch in range(im.shape[-1]):
            matched[..., ch] = _replace_lf(im[..., ch], ref[..., ch], lf_ratio)
    else:
        # Single channel
        matched = _replace_lf(im, ref, lf_ratio).astype(im.dtype)

    return matched


def inv_pca(im, joblib_path):
    """
    Perform inverse PCA transformation.

    Args:
        im (np.ndarray): Input image after performing PCA.
        joblib_path (str): Path of *.joblib file that stores PCA information.

    Returns:
        np.ndarray: Reconstructed input image.
    """

    pca = load(joblib_path)
    H, W, C = im.shape
    n_im = np.reshape(im, (-1, C))
    r_im = pca.inverse_transform(n_im)
    r_im = np.reshape(r_im, (H, W, -1))
    return r_im


def decode_seg_mask(mask_path):
    """
    Decode a segmentation mask image.
    
    Args:
        mask_path (str): Path of the mask image to decode.

    Returns:
        np.ndarray: Decoded mask image.
    """

    mask = np.asarray(Image.open(mask_path))
    mask = mask.astype('int64')
    return mask


def calc_hr_shape(lr_shape, sr_factor):
    return tuple(int(s * sr_factor) for s in lr_shape)
