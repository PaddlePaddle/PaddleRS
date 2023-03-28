# PaddleRS Data Transform Operator Construct Parameter

This document describes the parameters of each PaddleRS data transform operator in detail, including the operator name, operator purpose, parameter name, parameter type, parameter meaning, and parameter Default Value value of each operator.

## `AppendIndex`

Append remote sensing index to input image(s).

| Parameter Name             | Description                                                                                                                                        | Default Value       |
|-----------------|----------------------------------------------------------------------------------------------------------------------------------------------------|-----------|
|`index_type (str)`| Type of remote sensinng index. See supported index types in https://github.com/PaddlePaddle/PaddleRS/tree/develop/paddlers/transforms/indices.py . |           |
|`band_indexes (dict，optional)`|Mapping of band names to band indices (starting from 1)`. See band names in  https://github.com/PaddlePaddle/PaddleRS/tree/develop/paddlers/transforms/indices.py。                                           | `None`      |
|`satellite (str，optional)`|Type of satellite. If set, band indices will be automatically determined accordingly. See supported satellites in https://github.com/PaddlePaddle/PaddleRS/tree/develop/paddlers/transforms/satellites.py。                               | `None`      |


## `CenterCrop`

+ Crop the input image(s) at the center.
  - 1. Locate the center of the image.
  - 2. Crop the image.


| Parameter Name             | Description                                                                                                       | Default Value  |
|-----------------|----------------------------------------------------------------------------------------------------------|------|
|`crop_size (int, optional)`| Target size of the cropped image(s)  | `224`  |

## `Dehaze`

 Dehaze input image(s)


| Parameter Name             | Description                                   | Default Value   |
|-----------------|---------------------------------------------------|-------|
|`gamma (bool，optional)`| Use gamma correction or not  | `False` |

## `MatchRadiance`

Perform relative radiometric correction between bi-temporal images.

| Parameter Name             | Description                                                                                                                                                                                                                                                                 | Default Value |
|-----------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------|
|`method (str，optional)`| Method used to match the radiance of the bi-temporal images. Choices are {`'hist'`, `'lsr'`, `'fft'`}. `'hist'` stands for histogram matching, `'lsr'` stands for least-squares regression, and `'fft'` replaces the low-frequency components of the image to match the reference image. | `'hist'` |


## `MixupImage`

Mixup two images and their gt_bbbox/gt_score.

| Parameter Name             | Description                                     | Default Value |
|-----------------|-----------------------------------------------------|-----|
|`alpha (float，optional)`| Alpha parameter of beta distribution. | `1.5` |
|`beta (float，optional)` |Beta parameter of beta distribution. | `1.5` |

## `Normalize`

+ Apply normalization to the input image(s). The normalization steps are:

  - 1. im = (im - min_value) * 1 / (max_value - min_value)
  - 2. im = im - mean
  - 3. im = im / std


| Parameter Name      | Description                                                              | Default Value                          |
|---------------------|--------------------------------------------------------------------------|------------------------------|
| `mean (list[float] \| tuple[float]，optional)`  | Mean of input image(s)                                                   | `[0.485,0.456,0.406]` |
| `std (list[float] \| tuple[float]，optional)`   | Standard deviation of input image(s)                                     | `[0.229,0.224,0.225]` |
| `min_val (list[float] \| tuple[float]，optional)` | Inimum value of input image(s). If `None`, use `0` for all channels.     |    `None`      |
| `max_val (list[float] \| tuple[float]，optional)` | Maximum value of input image(s). If `None`, use `255`. for all channels. |  `None`        |
| `apply_to_tar (bool，optional)` \| Whether to apply transformation to the target image                      | `True`                         |

## `Pad`

Pad image to a specified size or multiple of `size_divisor`.

| Parameter Name           | Description                                                                                                                                                                                                          | Default Value              |
|--------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------|
| `target_size (list[int] \| tuple[int]，optional)`     | Image target size, if `None`, pad to multiple of size_divisor.                                                                                                                                                         | `None`               |
| `pad_mode (int，optional)` | Currently only four modes are supported:[-1, 0, 1, 2]. if `-1`, use specified offsets. If `0`, only pad to right and bottom If `1`, pad according to center. If `2`, only pad left and top.   | `0`                  |
| `offset (list[int] \| None，optional)`                |  Padding offsets.                                                                                                                                                                                                              | `None`               |
| `im_padding_value (list[float] \| tuple[float])` | RGB value of padded area.                                                                                                                                                                                                        | `（127.5,127.5,127.5)` |
| `label_padding_value (int，optional)` |Filling value for the mask.                                                                                                                                                                                                              | `255`                  |
| `size_divisor (int)`     | Image width and height after padding will be a multiple of `size_divisor`.                                                                                                                                                                       |                      |

## `RandomBlur`

Randomly blur input image(s).

| Parameter Name             | Description                                     | Default Value  |
|-----------------|-----------------------------------------------------|------|
|`probb (float)`|Probability of blurring. |      |

## `RandomCrop`

+ Randomly crop the input.

  - 1. Compute the height and width of cropped area according to `aspect_ratio` and
          `scaling`.
  - 2. Locate the upper left corner of cropped area randomly.
  - 3. Crop the image(s).
  - 4. Resize the cropped area to `crop_size` x `crop_size`.

| Parameter Name   | Description                                                                   | Default Value                     |
|------------------|-------------------------------------------------------------------------------|-------------------------|
| `crop_size (int \| list[int] \| tuple[int])` | Target size of the cropped area. If `None`, the cropped area will not be resized. | `None`                    |
| `aspect_ratio (list[float]，optional)` | Aspect ratio of cropped region in [min, max] format.                          | `[.5, 2.]`                |
| `thresholds (list[float]，optional)` | IoU thresholds to decide a valid bbox crop.                                   | `[.0,.1， .3， .5， .7， .9]` |
| `scaling (list[float], optional)` | Ratio between the cropped region and the original image in [min, max] format. | `[.3, 1.]`                |
| `num_attempts (int，optional)` | Max number of tries before giving up.                                         | `50`                      |
| `allow_no_crop (bool，optional)` | Whether returning without doing crop is allowed.                              | `True`                    |
| `cover_all_box (bool，optional)` | Whether to force to cover the entire target box.                              | `False`                   |

## `RandomDistort`

Random color distortion.

| Parameter Name                       | Description                                                     | Default Value   |
|----------------------------|-----------------------------------------------------------------|-------|
| `brightness_range (float，optional)` | Range of brightness distortion.                                 | `.5`    |
| `brightness_prob (float，optional)` | Probability of brightness distortion.                           | `.5`    |
| `contrast_range (float, optional)` | Range of contrast distortion.                                   | `.5`    |
| `contrast_prob (float, optional)` | Probability of contrast distortion.                             | `.5`    |
| `saturation_range (float,optional)` | Range of saturation distortion.                                 | `.5`    |
| `saturation_prob (float，optional)` | Probability of saturation distortion.                           | `.5`    |
| `hue_range (float，optional)` | Range of hue distortion.                                        | `.5`    |
| `hue_probb (float，optional)`| Probability of hue distortion.                                  | `.5`    |
| `random_apply (bool，optional)` | Apply the transformation in random (yolo) or fixed (SSD) order. | `True`  |
| `count (int，optional)`  | Count used to control the distortion                | `4`     |
| `shuffle_channel (bool，optional)` | Whether to swap channels randomly.                                           | `False` |


## `RandomExpand`

Randomly expand the input by padding according to random offsets.

| Parameter Name                  | Description                                    | Default Value                 |
|---------------------------------|----------------------------------------------------|---------------------|
| `upper_ratio (float，optional)`  | Maximum ratio to which the original image is expanded. | `4`                   |
| `probb (float，optional)`        |Probability of apply expanding. | `.5`                  |
| `im_padding_value (list[float] \| tuple[float]，optional)` |  RGB filling value for the image  | `(127.5,127.5,127.5)` |
| `label_padding_value (int，optional)` | Filling value for the mask.  | `255`    |

## `RandomHorizontalFlip`

Randomly flip the input horizontally.

| Parameter Name                                              | Description        | Default Value                |
|--------------------------------------------------|-----------|---------------------|
| `probb (float，optional)`                           | Probability of flipping the input   | `.5`                  |

## `RandomResize`

Resize input to random sizes.

+ Attention: If `interp` is 'RANDOM', the interpolation method will be chosen randomly.

| Parameter Name            | Description                                                          | Default Value                 |
|---------------------------|----------------------------------------------------------------------|---------------------|
| `Target_sizes (list[int] \| list[list \| tuple] \| tuple [list \| tuple])` | Multiple target sizes, each of which should be int, list, or tuple.  | `.5`                  |
| `interp (str，optional)`   | Interpolation method for resizing image(s). One of {`'NEAREST'`, `'LINEAR'`, `'CUBIC'`, `'AREA'`, `'LANCZOS4'`, `'RANDOM'`}. |   `'LINEAR'`                  ||


## `RandomResizeByShort`

Resize input to random sizes while keeping the aspect ratio.

+ Attention: If `interp` is 'RANDOM', the interpolation method will be chosen randomly.

| Parameter Name     | Description        | Default Value |
|--------------------|-----------|-----|
| `short_sizes (int \| list[int])` | Target size of the shorter side of the image(s).| `.5`  |
| `max_size (int，optional)` |Upper bound of longer side of the image(s). If `max_size` is -1, no upper bound will be applied.    | `-1`  |
| `interp (str，optional)` |  Interpolation method for resizing image(s). One of {'`NEAREST'`, `'LINEAR'`, `'CUBIC'`, `'AREA'`, `'LANCZOS4'`, `'RANDOM'`}.  | `'LINEAR'`    |

## `RandomScaleAspect`

Crop input image(s) and resize back to original sizes.


| Parameter Name                                                               | Description                                                                                          | Default Value    |
|-------------------------------------------------------------------|------------------------------------------------------------------------------------------------------|--------|
| `min_scale (float)`| Minimum ratio between the cropped region and the original image. If `0`, image(s) will not be cropped. | `0`      |
| `aspect_ratio (float)`    | Aspect ratio of cropped region.                                                                                 | `.33`    |

## `RandomSwap`

Randomly swap multi-temporal images.


| Parameter Name                                                               | Description        | Default Value |
|-------------------------------------------------------------------|-----------|-----|
|`probb (float，optional)`| Probability of swapping the input images.| `0.2` |

## `RandomVerticalFlip`
Randomly flip the input vertically.


| Parameter Name                                                              | Description        | Default Value |
|------------------------------------------------------------------|-----------|-----|
|`prob (float，optional)`| Probability of flipping the input| `.5`  |


## `ReduceDim`
Use PCA to reduce the dimension of input image(s).

| Parameter Name                                                               | Description                                          | Default Value  |
|-------------------------------------------------------------------|------------------------------------------------------|------|
|`joblib_path (str)`| Path of *.joblib file of PCA                         |      |
|`apply_to_tar (bool，optional)` | Whether to apply transformation to the target image. | `True` |


## `Resize`
Resize input.

    - If `target_size` is an int, resize the image(s) to (`target_size`, `target_size`).
    - If `target_size` is a list or tuple, resize the image(s) to `target_size`.
    Attention: If `interp` is 'RANDOM', the interpolation method will be chosen randomly.

| Parameter Name     | Description                                                                                                                                                          | Default Value      |
|--------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------|
| `target_size (int \| list[int] \| tuple[int])` | Target size. If it is an integer, the target height and width will be both set to `target_size`. Otherwise,  `target_size` represents [target height, target width]. |          |
| `interp (str，optional)` | Interpolation method for resizing image(s). One of {`'NEAREST'`, `'LINEAR'`, `'CUBIC'`, `'AREA'`, `'LANCZOS4'`, `'RANDOM'`}.                                         | `'LINEAR'` |
| `keep_ratio (bool，optional)` | If `True`, the scaling factor of width and height will be set to same value, and height/width of the resized image will be not  greater than the target width/height. | `False`    |

## `ResizeByLong`
Resize the input image, keeping the aspect ratio unchanged (calculate the scaling factor based on the long side).

    Attention: If `interp` is 'RANDOM', the interpolation method will be chosen randomly.


| Parameter Name                                        | Description        | Default Value      |
|--------------------------------------------|-----------|----------|
| `long_size (int)`|The size of the target on the longer side of the image.|          |
| `interp (str，optional)`                    | Interpolation method for resizing image(s). One of {`'NEAREST'`, `'LINEAR'`, `'CUBIC'`, `'AREA'`, `'LANCZOS4'`, `'RANDOM'`}.  | `'LINEAR'` |

## `ResizeByShort`
Resize input while keeping the aspect ratio.

    Attention: If `interp` is 'RANDOM', the interpolation method will be chosen randomly.


| Parameter Name              | Description                                                                                      | Default Value      |
|------------------|--------------------------------------------------------------------------------------------------|----------|
| `short_size (int)` | Target size of the shorter side of the image(s).                                                 |          |
| `mamax_size (int，optional)` | Upper bound of longer side of the image(s). If `max_size` is -1, no upper bound will be applied. | `-1`       |
| `interp (str，optional)`  | Interpolation method for resizing image(s). One of {`'NEAREST'`, `'LINEAR'`, `'CUBIC'`, `'AREA'`, `'LANCZOS4'`, `'RANDOM'`}.          | `'LINEAR'` |


## `SelectBand`
Select a set of bands of input image(s).

| Parameter Name              | Description                                          | Default Value      |
|------------------|------------------------------------------------------|----------|
| `band_list (list，optional)` | Bands to select (band index starts from 1).          | `[1,2,3]`  |
| `apply_to_tar (bool，optional)`| Whether to apply transformation to the target image. | `True`     |
