[简体中文](data_cn.md) | English

# Data Related APIs

## Dataset

In PaddleRS, all datasets inherit from the parent class [`BaseDataset`](https://github.com/PaddlePaddle/PaddleRS/blob/develop/paddlers/datasets/base.py).

### Change Detection Dataset `CDDataset`

`CDDataset` is defined in: https://github.com/PaddlePaddle/PaddleRS/blob/develop/paddlers/datasets/cd_dataset.py

The initialization parameter list is as follows:

|Parameter Name|Type|Parameter Description|Default Value|
|-------|----|--------|-----|
|`data_dir`|`str`|Directory that stores the dataset.||
|`file_list`|`str`|File list path. File list is a text file, in which each line contains the path infomation of one sample. The specific requirements of `CDDataset` on the file list are listed below.||
|`transforms`|`paddlers.transforms.Compose` \| `list`|Data transformation operators applied to input data.||
|`label_list`|`str` \| `None`|Label list path. Label list is a text file, in which each line contains the name of class.|`None`|
|`num_workers`|`int` \| `str`|Number of auxiliary processes used when loading data. If it is set to `'auto'`, use the following rules to determine the number of processes to use: When the number of CPU cores is greater than 16, 8 data read auxiliary processes are used; otherwise, the number of auxiliary processes is set to half the counts of CPU cores.|`'auto'`|
|`shuffle`|`bool`|Whether to randomly shuffle the samples in the dataset.|`False`|
|`with_seg_labels`|`bool`|Specify this option as `True` when the dataset contains segmentation labels for each phase.|`False`|
|`binarize_labels`|`bool`|If it is `True`, the change labels (and the segmentation label) are binarized after all data transformation operators are applied. For example, binarize labels valued in {0, 255} to {0, 1}.|`False`|

The requirements of `CDDataset` for the file list are as follows:

- If `with_seg_labels` is `False`, each line in the file list should contain three space-separated items representing, in turn, the path to the image of the first temporal phase, path to the image of the second temporal phase, and the path to the change label. Each given path should be the path relative to `data_dir`.
- If `with_seg_labels` is `True`, each line in the file list should contain five space-separated items, the first three of which have the same meaning as `with_seg_labels` is `False`, and the last two represent the path of the segmentation labels for the first and second phase images (also in relative paths `data_dir`).

### Scenario Classification Dataset `ClasDataset`

`ClasDataset` is defined in: https://github.com/PaddlePaddle/PaddleRS/blob/develop/paddlers/datasets/clas_dataset.py

The initialization parameter list is as follows:

|Parameter Name|Type|Parameter Description|Default Value|
|-------|----|--------|-----|
|`data_dir`|`str`|Directory that stores the dataset.||
|`file_list`|`str`|File list path. File list is a text file, in which each line contains the path infomation of one sample.The specific requirements of `ClasDataset` on the file list are listed below.||
|`transforms`|`paddlers.transforms.Compose` \| `list`|Data transformation operators applied to input data.||
|`label_list`|`str` \| `None`|Label list path. Label list is a text file, in which each line contains the name of class.|`None`|
|`num_workers`|`int` \| `str`|Number of auxiliary processes used when loading data. If it is set to `'auto'`, use the following rules to determine the number of processes to use: When the number of CPU cores is greater than 16, 8 data read auxiliary processes are used; otherwise, the number of auxiliary processes is set to half the counts of CPU cores.|`'auto'`|
|`shuffle`|`bool`|Whether to randomly shuffle the samples in the dataset.|`False`|

The requirements of `ClasDataset` for the file list are as follows:

- Each line in the file list should contain two space-separated items representing, in turn, the path of input image relative to `data_dir` and the category ID of the image (which can be parsed as an integer value).

### COCO Format Object Detection Dataset `COCODetDataset`

`COCODetDataset` is defined in: https://github.com/PaddlePaddle/PaddleRS/blob/develop/paddlers/datasets/coco.py

The initialization parameter list is as follows:

|Parameter Name|Type|Parameter Description|Default Value|
|-------|----|--------|-----|
|`data_dir`|`str`|Directory that stores the dataset.||
|`image_dir`|`str`|Directory of input images.||
|`anno_path`|`str`|[COCO Format](https://cocodataset.org/#home)label file path.||
|`transforms`|`paddlers.transforms.Compose` \| `list`|Data transformation operators applied to input data.||
|`label_list`|`str` \| `None`|Label list path. Label list is a text file, in which each line contains the name of class.|`None`|
|`num_workers`|`int` \| `str`|Number of auxiliary processes used when loading data. If it is set to `'auto'`, use the following rules to determine the number of processes to use: When the number of CPU cores is greater than 16, 8 data read auxiliary processes are used; otherwise, the number of auxiliary processes is set to half the counts of CPU cores.|`'auto'`|
|`shuffle`|`bool`|Whether to randomly shuffle the samples in the dataset.|`False`|
|`allow_empty`|`bool`|Whether to add negative samples to the dataset.|`False`|
|`empty_ratio`|`float`|Negative sample ratio. Take effect only if `allow_empty` is `True`. If `empty_ratio` is negative or greater than or equal to 1, all negative samples generated are retained.|`1.0`|
|`batch_transforms`|`paddlers.transforms.BatchCompose` \| `list`|Data batch transformation operators applied to input data.||

### VOC Format Object Detection Dataset `VOCDetDataset`

`VOCDetDataset` is defined in: https://github.com/PaddlePaddle/PaddleRS/blob/develop/paddlers/datasets/voc.py

The initialization parameter list is as follows:

|Parameter Name|Type|Parameter Description|Default Value|
|-------|----|--------|-----|
|`data_dir`|`str`|Directory that stores the dataset. ||
|`file_list`|`str`|File list path. File list is a text file, in which each line contains the path infomation of one sample.The specific requirements of `VOCDetDataset` on the file list are listed below.||
|`transforms`|`paddlers.transforms.Compose` \| `list`|Data transformation operators applied to input data. ||
|`label_list`|`str` \| `None`|Label list path. Label list is a text file, in which each line contains the name of class.|`None`|
|`num_workers`|`int` \| `str`|Number of auxiliary processes used when loading data. If it is set to `'auto'`, use the following rules to determine the number of processes to use: When the number of CPU cores is greater than 16, 8 data read auxiliary processes are used; otherwise, the number of auxiliary processes is set to half the counts of CPU cores.|`'auto'`|
|`shuffle`|`bool`|Whether to randomly shuffle the samples in the dataset.|`False`|
|`allow_empty`|`bool`|Whether to add negative samples to the dataset.|`False`|
|`empty_ratio`|`float`|Negative sample ratio. Takes effect only if `allow_empty` is `True`. If `empty_ratio` is negative or greater than or equal to `1`, all negative samples generated will be retained.|`1.0`|
|`batch_transforms`|`paddlers.transforms.BatchCompose` \| `list`|Data batch transformation operators applied to input data.||

The requirements of `VOCDetDataset` for the file list are as follows:

- Each line in the file list should contain two space-separated items representing, in turn, the path of input image relative to `data_dir` and the path of [Pascal VOC Format](http://host.robots.ox.ac.uk/pascal/VOC/) label file relative to `data_dir`.

### Image Restoration Dataset `ResDataset`

`ResDataset` is defined in: https://github.com/PaddlePaddle/PaddleRS/blob/develop/paddlers/datasets/res_dataset.py

The initialization parameter list is as follows:

|Parameter Name|Type|Parameter Description|Default Value|
|-------|----|--------|-----|
|`data_dir`|`str`|Directory that stores the dataset.||
|`file_list`|`str`|File list path. file list is a text file, in which each line contains the path infomation of one sample.The specific requirements of `ResDataset` on the file list are listed below.||
|`transforms`|`paddlers.transforms.Compose` \| `list`|Data transformation operators applied to input data.||
|`num_workers`|`int` \| `str`|Number of auxiliary processes used when loading data. If it is set to `'auto'`, use the following rules to determine the number of processes to use: When the number of CPU cores is greater than 16, 8 data read auxiliary processes are used; otherwise, the number of auxiliary processes is set to half the counts of CPU cores.|`'auto'`|
|`shuffle`|`bool`|Whether to randomly shuffle the samples in the dataset.|`False`|
|`sr_factor`|`int` \| `None`|For super resolution reconstruction task, this is the scaling factor. For other tasks, please specify `sr_factor` as `None`.|`None`|

The requirements of `ResDataset` for the file list are as follows:

- Each line in the file list should contain two space-separated items representing, in turn, representing the path of the input image (such as a low-resolution image in a super-resolution reconstruction task) relative to the `data_dir` and the path of the target image (such as a high-resolution image in a super-resolution reconstruction task) relative to the `data_dir`.

### Image Segmentation Dataset `SegDataset`

`SegDataset` is defined in: https://github.com/PaddlePaddle/PaddleRS/blob/develop/paddlers/datasets/seg_dataset.py

The initialization parameter list is as follows:

|Parameter Name|Type|Parameter Description|Default Value|
|-------|----|--------|-----|
|`data_dir`|`str`|Directory that stores the dataset.||
|`file_list`|`str`|File list path. file list is a text file, in which each line contains the path infomation of one sample.The specific requirements of `SegDataset` on the file list are listed below.||
|`transforms`|`paddlers.transforms.Compose` \| `list`|Data transformation operators applied to input data.||
|`label_list`|`str` \| `None`|Label list path. Label list is a text file, in which each line contains the name of class.|`None`|
|`num_workers`|`int` \| `str`|Number of auxiliary processes used when loading data. If it is set to `'auto'`, use the following rules to determine the number of processes to use: When the number of CPU cores is greater than 16, 8 data read auxiliary processes are used; otherwise, the number of auxiliary processes is set to half the counts of CPU cores.|`'auto'`|
|`shuffle`|`bool`|Whether to randomly shuffle the samples in the dataset.|`False`|

The requirements of `SegDataset` for the file list are as follows:

- Each line in the file list should contain two space-separated items representing, in turn, the path of input image relative to `data_dir` and the path of the segmentation label relative to `data_dir`.

## API of Data Reading

Remote sensing images come from various sources and their data formats are very complicated. PaddleRS provides a unified interface for reading remote sensing images of different types and formats. At present, PaddleRS can read common file formats such as .png, .jpg, .bmp, and .npy, as well as handle GeoTiff, img, and other image formats commonly used in remote sensing.

Depending on the practical demands, the user can choose `paddlers.transforms.decode_image()` or `paddlers.transforms.DecodeImg` to read data. `DecodeImg` is one of [data transformation operators](#data-transformation-operators), can be combined with other operators. `decode_image` is the encapsulation of `DecodeImg` operator, which is convenient use in the way of function calls.

The parameter list of `decode_image()` function is as follows:

|Parameter Name|Type|Parameter Description|Default Value|
|-------|----|--------|-----|
|`im_path`|`str`|Path of input image.||
|`to_rgb`|`bool`|If `True`, the conversion of BGR to RGB format is performed. This parameter is not used and may be removed in the future. Do not use it if possible.|`True`|
|`to_uint8`|`bool`|If `True`, the image data read is quantized and converted to uint8.|`True`|
|`decode_bgr`|`bool`|If `True`, automatically parses non-geo format images (such as jpeg images) into BGR format.|`True`|
|`decode_sar`|`bool`|If `True`, single-channel geo-format images (such as GeoTiff images) are automatically parsed as SAR images.|`True`|
|`read_geo_info`|`bool`|If `True`, the geographic information is read from the image.|`False`|
|`use_stretch`|`bool`|Whether to apply a linear stretch to image image brightness (with 2% max and min values removed). Take effect only if `to_uint8` is `True`.|`False`|
|`read_raw`|`bool`|If `True`, it is equivalent to specifying `to_rgb` as `True` and `to_uint8` as `False`, and this parameter has a higher priority than the above.|`False`|

The return format is as follows:

- If `read_geo_info` is `False`, the image ([h, w, c] arrangement) is returned in the format of `numpy.ndarray`.
- If `read_geo_info` is `True`, return a tuple consisting of two elements. The first element is the image data, and the second element is a dictionary containing the geographic information of the image, such as the geotransform information and geographic projection information.

## Data Transformation Operators

In PaddleRS a series of classes are defined that, when instantiated, perform certain data preprocessing or data augmentation operations by calling the `__call__` method. PaddleRS calls these classes data preprocessing/data augmentation operators, and collectively **Data Transform Operators**. All data transformation operators inherit from the parent class[`Transform`](https://github.com/PaddlePaddle/PaddleRS/blob/develop/paddlers/transforms/operators.py).

### `Transform`

The `__call__` method of the `Transform` object takes a unique argument `sample`. `sample` must be a dictionary or a sequence of dictionaries. When `sample` is a sequence, the `Transform` object performs data transformations for each dictionary in `sample` and returns the results sequentially stored in a Python build-in list. When `sample` is a dictionary, the `Transform` object extracts input from some of its key-value pairs (these keys are called "input keys"), performs the transformation, and writes the results as key-value pairs into `sample` (these keys are called *output keys*). It should be noted that many of the `Transform` objects in PaddleRS overwrite key-value pairs, that is, there is an intersection between the input key and the output key. The common keys in `sample` and their meanings are as follows:

|Key Name|Description|
|----|----|
|`'image'`|Image path or data. For change detection tasks, it refers to the first image.|
|`'image2'`|Second image in change detection tasks.|
|`'image_t1'`|Path of th first image in change detection tasks.|
|`'image_t2'`|Path of the second image in change detection tasks.|
|`'mask'`|Ground-truth label path or data in image segmentation/change detection tasks.|
|`'aux_masks'`|Auxiliary label path or data in image segmentation/change detection tasks.|
|`'gt_bbox'`|Bounding box annotations in object detection tasks.|
|`'gt_poly'`|Polygon annotations in object detection tasks.|
|`'target'`|Target image path or data in image restoration tasks.|

### Construct Data Transformation Operators

Please refer to [this document](../intro/transforms_cons_params_en.md).

### Combine Data Transformation Operators

Use `paddlers.transforms.Compose` to combine a set of data transformation operators. `Compose` receives a list input when constructed. When you call `Compose`, it serially execute each data transform operators in the list. The following is an example:

```python
# Compose a variety of transformations using Compose.
# The transformations contained in Compose will be executed sequentially in sequence
train_transforms = T.Compose([
    # Scale the image to 512x512
    T.Resize(target_size=512),
    # Perform a random horizontal flip with a 50% probability
    T.RandomHorizontalFlip(prob=0.5),
    # Normalize data to [-1,1]
    T.Normalize(
        mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
```

For the validation dataset of image segmentation task and change detection task, a `ReloadMask` operator can be added to reload the ground-truth label. The following is an example:

```python
eval_transforms = T.Compose([
    T.Resize(target_size=512),
    T.Normalize(
        mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    # Reload label
    T.ReloadMask()
])
```
