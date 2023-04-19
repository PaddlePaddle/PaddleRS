[简体中文](transforms_cons_params_cn.md) | English

# PaddleRS Data Transformation Operator Construction Parameters

This document describes the parameters of each PaddleRS data transformation operator, including the operator name, operator purpose, parameter name, parameter type, parameter meaning, and parameter default value of each operator.

You can check all data transformation operators supported by PaddleRS [here](../intro/transforms_en.md).

## `AppendIndex`

Append remote sensing index to input image(s).

| Parameter Name (Parameter Type)             | Description                                                                                                                                        | Default Value       |
|-----------------|----------------------------------------------------------------------------------------------------------------------------------------------------|-----------|
|`index_type (str)`| Type of remote sensinng index. See supported index types in https://github.com/PaddlePaddle/PaddleRS/tree/develop/paddlers/transforms/indices.py |           |
|`band_indexes (dict)`|Mapping of band names to band indices (starting from 1). See supported band names in  https://github.com/PaddlePaddle/PaddleRS/tree/develop/paddlers/transforms/indices.py                                         | `None`      |
|`satellite (str)`|Type of satellite. If set, band indices will be automatically determined accordingly. See supported satellites in https://github.com/PaddlePaddle/PaddleRS/tree/develop/paddlers/transforms/satellites.py                             | `None`      |


## `CenterCrop`

Crop the input image(s) at the center.

1. Locate the center of the image.
2. Crop the image.

| Parameter Name (Parameter Type)             | Description                                                                                                       | Default Value  |
|-----------------|----------------------------------------------------------------------------------------------------------|------|
|`crop_size (int)`| Target size of the cropped image(s)  | `224`  |


## `Dehaze`

Dehaze input image(s)

| Parameter Name (Parameter Type)             | Description                                   | Default Value   |
|-----------------|---------------------------------------------------|-------|
|`gamma (bool)`| Use gamma correction or not  | `False` |


## `MatchRadiance`

Perform relative radiometric correction between bi-temporal images.

| Parameter Name (Parameter Type)             | Description                                                                                                                                                                                                                                                                 | Default Value |
|-----------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------|
|`method (str)`| Method used to match the radiance of the bi-temporal images. Choices are {`'hist'`, `'lsr'`, `'fft'`}. `'hist'` stands for histogram matching, `'lsr'` stands for least-squares regression, and `'fft'` replaces the low-frequency components of the image to match the reference image | `'hist'` |


## `MixupImage`

Mixup two images and their gt_bbox/gt_score.

| Parameter Name (Parameter Type)             | Description                                     | Default Value |
|-----------------|-----------------------------------------------------|-----|
|`alpha (float)`|Alpha parameter of beta distribution | `1.5` |
|`beta (float)` |Beta parameter of beta distribution | `1.5` |


## `Normalize`

Apply normalization to the input image(s). The normalization steps are:

1. im = (im - min_value) * 1 / (max_value - min_value)
2. im = im - mean
3. im = im / std

| Parameter Name (Parameter Type)      | Description                                                              | Default Value                          |
|---------------------|--------------------------------------------------------------------------|------------------------------|
| `mean (list[float] \| tuple[float])`  | Mean of input image(s)                                                   | `[0.485,0.456,0.406]` |
| `std (list[float] \| tuple[float])`   | Standard deviation of input image(s)                                     | `[0.229,0.224,0.225]` |
| `min_val (list[float] \| tuple[float])` | Inimum value of input image(s). If `None`, use `0` for all channels     |    `None`      |
| `max_val (list[float] \| tuple[float])` | Maximum value of input image(s). If `None`, use `255` for all channels |  `None`        |
| `apply_to_tar (bool)` \| Whether to apply transformation to the target image                      | `True`                         |


## `Pad`

Pad image to a specified size or multiple of `size_divisor`.

| Parameter Name (Parameter Type)           | Description                                                                                                                                                                                                          | Default Value              |
|--------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------|
| `target_size (list[int] \| tuple[int])`     | Image target size, if `None`, pad to multiple of size_divisor                                                                                                                                                        | `None`               |
| `pad_mode (int)` | Currently only four modes are supported:[-1, 0, 1, 2]. if `-1`, use specified offsets. If `0`, only pad to right and bottom. If `1`, pad according to center. If `2`, only pad left and top   | `0`                  |
| `offset (list[int] \| None)`                |  Padding offsets                                                                                                                                                                                                              | `None`               |
| `im_padding_value (list[float] \| tuple[float])` | RGB value of padded area                                                                                                                                                                                                        | `(127.5,127.5,127.5)` |
| `label_padding_value (int)` |Filling value for the mask                                                                                                                                                                                                              | `255`                  |
| `size_divisor (int)`     | Image width and height after padding will be a multiple of `size_divisor`                                                                                                                                                                       |                      |


## `RandomBlur`

Randomly blur input image(s).

| Parameter Name (Parameter Type)             | Description                                     | Default Value  |
|-----------------|-----------------------------------------------------|------|
|`prob (float)`|Probability of blurring |      |


## `RandomCrop`

Randomly crop the input.

1. Compute the height and width of cropped area according to `aspect_ratio` and`scaling`.
2. Locate the upper left corner of cropped area randomly.
3. Crop the image(s).
4. Resize the cropped area to `crop_size` x `crop_size`.

| Parameter Name (Parameter Type)   | Description                                                                   | Default Value                     |
|------------------|-------------------------------------------------------------------------------|-------------------------|
| `crop_size (int \| list[int] \| tuple[int])` | Target size of the cropped area. If `None`, the cropped area will not be resized | `None`                    |
| `aspect_ratio (list[float])` | Aspect ratio of cropped region in [min, max] format                          | `[.5, 2.]`                |
| `thresholds (list[float])` | IoU thresholds to decide a valid bbox crop                                   | `[.0,.1,  .3,  .5,  .7,  .9]` |
| `scaling (list[float])` | Ratio between the cropped region and the original image in [min, max] format | `[.3, 1.]`                |
| `num_attempts (int)` | Max number of tries before giving up                                         | `50`                      |
| `allow_no_crop (bool)` | Whether returning without doing crop is allowed                              | `True`                    |
| `cover_all_box (bool)` | Whether to force to cover the entire target box                              | `False`                   |


## `RandomDistort`

Random color distortion.

| Parameter Name (Parameter Type)                       | Description                                                     | Default Value   |
|----------------------------|-----------------------------------------------------------------|-------|
| `brightness_range (float)` | Range of brightness distortion                                 | `.5`    |
| `brightness_prob (float)` | Probability of brightness distortion                           | `.5`    |
| `contrast_range (float)` | Range of contrast distortion                                   | `.5`    |
| `contrast_prob (float)` | Probability of contrast distortion                             | `.5`    |
| `saturation_range (float,optional)` | Range of saturation distortion                                 | `.5`    |
| `saturation_prob (float)` | Probability of saturation distortion                           | `.5`    |
| `hue_range (float)` | Range of hue distortion                                        | `.5`    |
| `hue_prob (float)`| Probability of hue distortion                                  | `.5`    |
| `random_apply (bool)` | Apply the transformation in random (YOLO) or fixed (SSD) order | `True`  |
| `count (int)`  | Count used to control the distortion                | `4`     |
| `shuffle_channel (bool)` | Whether to permute channels randomly                                           | `False` |


## `RandomExpand`

Randomly expand the input by padding according to random offsets.

| Parameter Name (Parameter Type)                  | Description                                    | Default Value                 |
|---------------------------------|----------------------------------------------------|---------------------|
| `upper_ratio (float)`  | Maximum ratio to which the original image is expanded | `4`                   |
| `prob (float)`        |Probability of expanding | `.5`                  |
| `im_padding_value (list[float] \| tuple[float])` |  RGB filling value for the image  | `(127.5,127.5,127.5)` |
| `label_padding_value (int)` | Filling value for the mask  | `255`    |


## `RandomHorizontalFlip`

Randomly flip the input horizontally.

| Parameter Name (Parameter Type)                                              | Description        | Default Value                |
|--------------------------------------------------|-----------|---------------------|
| `prob (float)`                           | Probability of flipping the input   | `.5`                  |


## `RandomResize`

Resize input to random sizes.

Attention: If `interp` is 'RANDOM', the interpolation method will be chosen randomly.

| Parameter Name (Parameter Type)            | Description                                                          | Default Value                 |
|---------------------------|----------------------------------------------------------------------|---------------------|
| `Target_sizes (list[int] \| list[list \| tuple] \| tuple [list \| tuple])` | Multiple target sizes, each of which should be int, list, or tuple  | `.5`                  |
| `interp (str)`   | Interpolation method for resizing image(s). One of {`'NEAREST'`, `'LINEAR'`, `'CUBIC'`, `'AREA'`, `'LANCZOS4'`, `'RANDOM'`} |   `'LINEAR'`                  ||


## `RandomResizeByShort`

Resize input to random sizes while keeping the aspect ratio.

Attention: If `interp` is `'RANDOM'`, the interpolation method will be chosen randomly.

| Parameter Name (Parameter Type)     | Description        | Default Value |
|--------------------|-----------|-----|
| `short_sizes (int \| list[int])` | Target size of the shorter side of the image(s)| `.5`  |
| `max_size (int)` |Upper bound of longer side of the image(s). If `max_size` is -1, no upper bound will be applied    | `-1`  |
| `interp (str)` |  Interpolation method for resizing image(s). One of {'`NEAREST'`, `'LINEAR'`, `'CUBIC'`, `'AREA'`, `'LANCZOS4'`, `'RANDOM'`}  | `'LINEAR'`    |


## `RandomScaleAspect`

Crop input image(s) and resize back to original sizes.

| Parameter Name (Parameter Type)                                                               | Description                                                                                          | Default Value    |
|-------------------------------------------------------------------|------------------------------------------------------------------------------------------------------|--------|
| `min_scale (float)`| Minimum ratio between the cropped region and the original image. If `0`, image(s) will not be cropped | `0`      |
| `aspect_ratio (float)`    | Aspect ratio of cropped region                                                                                 | `.33`    |


## `RandomSwap`

Randomly swap multi-temporal images.

| Parameter Name (Parameter Type)                                                               | Description        | Default Value |
|-------------------------------------------------------------------|-----------|-----|
|`prob (float)`| Probability of swapping the input images| `0.2` |


## `RandomVerticalFlip`

Randomly flip the input vertically.

| Parameter Name (Parameter Type)                                                              | Description        | Default Value |
|------------------------------------------------------------------|-----------|-----|
|`prob (float)`| Probability of flipping the input| `.5`  |


## `ReduceDim`

Use PCA to reduce the dimension of input image(s).

| Parameter Name (Parameter Type)                                                               | Description                                          | Default Value  |
|-------------------------------------------------------------------|------------------------------------------------------|------|
|`joblib_path (str)`| Path of *.joblib file of PCA                         |      |
|`apply_to_tar (bool)` | Whether to apply transformation to the target image | `True` |


## `Resize`

Resize input.

- If `target_size` is an int, resize the image(s) to `target_size` x `target_size`.
- If `target_size` is a list or tuple, resize the image(s) to `target_size`.

Attention: If `interp` is `'RANDOM'`, the interpolation method will be chosen randomly.

| Parameter Name (Parameter Type)     | Description                                                                                                                                                          | Default Value      |
|--------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------|
| `target_size (int \| list[int] \| tuple[int])` | Target size. If it is an integer, the target height and width will be both set to `target_size`. Otherwise,  `target_size` represents [target height, target width] |          |
| `interp (str)` | Interpolation method for resizing image(s). One of {`'NEAREST'`, `'LINEAR'`, `'CUBIC'`, `'AREA'`, `'LANCZOS4'`, `'RANDOM'`}                                         | `'LINEAR'` |
| `keep_ratio (bool)` | If `True`, the scaling factor of width and height will be set to same value, and height/width of the resized image will be no greater than the target width/height | `False`    |


## `ResizeByLong`

Resize the input image, keeping the aspect ratio unchanged (calculate the scaling factor based on the long side).

Attention: If `interp` is `'RANDOM'`, the interpolation method will be chosen randomly.

| Parameter Name (Parameter Type)                                        | Description        | Default Value      |
|--------------------------------------------|-----------|----------|
| `long_size (int)`|The size of the target on the longer side of the image|          |
| `interp (str)`                    | Interpolation method for resizing image(s). One of {`'NEAREST'`, `'LINEAR'`, `'CUBIC'`, `'AREA'`, `'LANCZOS4'`, `'RANDOM'`}  | `'LINEAR'` |


## `ResizeByShort`

Resize input while keeping the aspect ratio.

Attention: If `interp` is `'RANDOM'`, the interpolation method will be chosen randomly.

| Parameter Name (Parameter Type)              | Description                                                                                      | Default Value      |
|------------------|--------------------------------------------------------------------------------------------------|----------|
| `short_size (int)` | Target size of the shorter side of the image(s)                                                 |          |
| `max_size (int)` | Upper bound of longer side of the image(s). If `max_size` is -1, no upper bound will be applied | `-1`       |
| `interp (str)`  | Interpolation method for resizing image(s). One of {`'NEAREST'`, `'LINEAR'`, `'CUBIC'`, `'AREA'`, `'LANCZOS4'`, `'RANDOM'`}          | `'LINEAR'` |


## `SelectBand`

Select a set of bands of input image(s).

| Parameter Name (Parameter Type)              | Description                                          | Default Value      |
|------------------|------------------------------------------------------|----------|
| `band_list (list)` | Bands to select (band index starts from 1)          | `[1, 2, 3]`  |
| `apply_to_tar (bool)`| Whether to apply transformation to the target image | `True`     |
