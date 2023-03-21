# Data related API description

## Dataset

In PaddleRS, all datasets inherit from the parent class [`BaseDataset`](https://github.com/PaddlePaddle/PaddleRS/blob/develop/paddlers/datasets/base.py).

### Change Detection Dataset`CDDataset`

`CDDataset` is defined in: https://github.com/PaddlePaddle/PaddleRS/blob/develop/paddlers/datasets/cd_dataset.py

The initialization parameter list is as follows:

|Parameter Name|Type|Parameter Description|Default Value|
|-------|----|--------|-----|
|`data_dir`|`str`|Dataset storage directory.||
|`file_list`|`str`|file list path. file list is a text file, in which each line contains the path infomation of sample.The specific requirements of `CDDataset` on the file list are listed below.||
|`transforms`|`paddlers.transforms.Compose`|Data transformation operator applied to input data.||
|`label_list`|`str` \| `None`|label list file. label list is a text file, in which each line contains the name of class.|`None`|
|`num_workers`|`int` \| `str`|The number of auxiliary processes used when loading data. If it is set to `'auto'`, the following rules determine the number of processes to use:When the number of CPU cores is greater than 16, 8 data read auxiliary processes are used; Otherwise, use CPU cores to count half as many auxiliary processes.|`'auto'`|
|`shuffle`|`bool`|Whether to randomly shuffle the samples in the dataset.|`False`|
|`with_seg_labels`|`bool`|Specify this option as `True` when the dataset contains segmentation labels for each phase.|`False`|
|`binarize_labels`|`bool`|If it is `True`, the change labels (and segmentation label) is binarized after all data transformation operators except `Arrange` are processed. For example, binarize a tag with the value {0, 255} to {0, 1}.|`False`|

The requirements of `CDDataset` for the file list are as follows:

- If `with_seg_labels` is `False`, each line in the file list should contain three space-separated items representing, in turn, the path to `data_dir` for the first phase image, `data_dir` for the second phase image, and the path to `data_dir` for the change label.
- If `with_seg_labels` is `True`, each line in the file list should contain five space-separated items, the first three of which have the same meaning as `with_seg_labels` is `False`, and the last two represent the path of the first and second phase images corresponding to the segmentation label `data_dir`.

### Scenario Classification Dataset`ClasDataset`

`ClasDataset` is define in: https://github.com/PaddlePaddle/PaddleRS/blob/develop/paddlers/datasets/clas_dataset.py

The initialization parameter list is as follows:

|Parameter Name|Type|Parameter Description|Default Value|
|-------|----|--------|-----|
|`data_dir`|`str`|Dataset storage directory.||
|`file_list`|`str`|file list path. file list is a text file, in which each line contains the path infomation of sample.The specific requirements of `ClasDataset` on the file list are listed below.||
|`transforms`|`paddlers.transforms.Compose`|Data transformation operator applied to input data.||
|`label_list`|`str` \| `None`|label list file. label list is a text file, in which each line contains the name of class.|`None`|
|`num_workers`|`int` \| `str`|The number of auxiliary processes used when loading data. If it is set to `'auto'`, the following rules determine the number of processes to use:When the number of CPU cores is greater than 16, 8 data read auxiliary processes are used; Otherwise, use CPU cores to count half as many auxiliary processes.|`'auto'`|
|`shuffle`|`bool`|Whether to randomly shuffle the samples in the dataset.|`False`|

The requirments of `ClasDataset` for the file list are as follows:

- Each line in the file list should contain two space-separated items representing, in turn, the path of input image relative to `data_dir` and the category ID of the image (which can be parsed as an integer value).

### COCO Format Object Detection Dataset`COCODetDataset`

`COCODetDataset` is define in: https://github.com/PaddlePaddle/PaddleRS/blob/develop/paddlers/datasets/coco.py

The initialization parameter list is as follows:

|Parameter Name|Type|Parameter Description|Default Value|
|-------|----|--------|-----|
|`data_dir`|`str`|Dataset storage directory.||
|`image_dir`|`str`|The directory of input images.||
|`ann_path`|`str`|[COCO Format](https://cocodataset.org/#home)label file path.||
|`transforms`|`paddlers.transforms.Compose`|Data transformation operator applied to input data.||
|`label_list`|`str` \| `None`|label list file. label list is a text file, in which each line contains the name of class.|`None`|
|`num_workers`|`int` \| `str`|The number of auxiliary processes used when loading data. If it is set to `'auto'`, the following rules determine the number of processes to use:When the number of CPU cores is greater than 16, 8 data read auxiliary processes are used; Otherwise, use CPU cores to count half as many auxiliary processes.|`'auto'`|
|`shuffle`|`bool`|Whether to randomly shuffle the samples in the dataset.|`False`|
|`allow_empty`|`bool`|Whether to add negative samples to the dataset.|`False`|
|`empty_ratio`|`float`|Negative sample ratio, take effect only if `allow_empty` is `True`. If `empty_ratio` is negative or greater than or equal to 1, all negative samples generated are retained.|`1.0`|

### VOC Format Object Detection Dataset`VOCDetDataset`

`VOCDetDataset` is define in: https://github.com/PaddlePaddle/PaddleRS/blob/develop/paddlers/datasets/voc.py

The initialization parameter list is as follows:

|Parameter Name|Type|Parameter Description|Default Value|
|-------|----|--------|-----|
|`data_dir`|`str`|Dataset storage directory.||
|`file_list`|`str`|file list path. file list is a text file, in which each line contains the path infomation of sample.The specific requirements of `VOCDetDataset` on the file list are listed below.||
|`transforms`|`paddlers.transforms.Compose`|Data transformation operator applied to input data.||
|`label_list`|`str` \| `None`|label list file. label list is a text file, in which each line contains the name of class.|`None`|
|`num_workers`|`int` \| `str`|The number of auxiliary processes used when loading data. If it is set to `'auto'`, the following rules determine the number of processes to use:When the number of CPU cores is greater than 16, 8 data read auxiliary processes are used; Otherwise, use CPU cores to count half as many auxiliary processes.|`'auto'`|
|`shuffle`|`bool`|Whether to randomly shuffle the samples in the dataset.|`False`|
|`allow_empty`|`bool`|Whether to add negative samples to the dataset.|`False`|
|`empty_ratio`|`float`|Negative sample ratio, take effect only if `allow_empty` is `True`. If `empty_ratio` is negative or greater than or equal to 1, all negative samples generated are retained.|`1.0`|

The requirments of `VOCDetDataset` for the file list are as follows:

- Each line in the file list should contain two space-separated items representing, in turn, the path of input image relative to `data_dir` and the path of [Pascal VOC Format](http://host.robots.ox.ac.uk/pascal/VOC/)label file relative to `data_dir`.

### Image Restoration Dataset`ResDataset`

`ResDataset` is define in: https://github.com/PaddlePaddle/PaddleRS/blob/develop/paddlers/datasets/res_dataset.py

The initialization parameter list is as follows:

|Parameter Name|Type|Parameter Description|Default Value|
|-------|----|--------|-----|
|`data_dir`|`str`|Dataset storage directory.||
|`file_list`|`str`|file list path. file list is a text file, in which each line contains the path infomation of sample.The specific requirements of `ResDataset` on the file list are listed below.||
|`transforms`|`paddlers.transforms.Compose`|Data transformation operator applied to input data.||
|`num_workers`|`int` \| `str`|The number of auxiliary processes used when loading data. If it is set to `'auto'`, the following rules determine the number of processes to use:When the number of CPU cores is greater than 16, 8 data read auxiliary processes are used; Otherwise, use CPU cores to count half as many auxiliary processes.|`'auto'`|
|`shuffle`|`bool`|Whether to randomly shuffle the samples in the dataset.|`False`|
|`sr_factor`|`int` \| `None`|For super resolution reconstruction task, specify as super resolution multiple; For other tasks, specify as `None`.|`None`|

The requirments of `ResDataset` for the file list are as follows:

- Each line in the file list should contain two space-separated items representing, in turn, representing The path of the input image (such as a low-resolution image in a super-resolution reconstruction task) relative to the `data_dir` and the path of the target image (such as a high-resolution image in a super-resolution reconstruction task) relative to the `data_dir`.

### Image Segmentation Dataset`SegDataset`

`SegDataset` is defined in: https://github.com/PaddlePaddle/PaddleRS/blob/develop/paddlers/datasets/seg_dataset.py

The initialization parameter list is as follows:

|Parameter Name|Type|Parameter Description|Default Value|
|-------|----|--------|-----|
|`data_dir`|`str`|Dataset storage directory.||
|`file_list`|`str`|file list path. file list is a text file, in which each line contains the path infomation of sample.The specific requirements of `SegDataset` on the file list are listed below.||
|`transforms`|`paddlers.transforms.Compose`|Data transformation operator applied to input data.||
|`label_list`|`str` \| `None`|label list file. label list is a text file, in which each line contains the name of class.|`None`|
|`num_workers`|`int` \| `str`|The number of auxiliary processes used when loading data. If it is set to `'auto'`, the following rules determine the number of processes to use:When the number of CPU cores is greater than 16, 8 data read auxiliary processes are used; Otherwise, use CPU cores to count half as many auxiliary processes.|`'auto'`|
|`shuffle`|`bool`|Whether to randomly shuffle the samples in the dataset.|`False`|

The requirments of `SegDataset` for the file list are as follows:

- Each line in the file list should contain two space-separated items representing, in turn, the path of input image relative to `data_dir` and the path of the segmentation label relative to `data_dir`.

## API of Data Reading

Remote sensing images come from various sources and their data formats are very complicated. PaddleRS provides a unified interface for reading remote sensing images of different types and formats. At present, PaddleRS can read common file formats such as .png, .jpg, .bmp and .npy, as well as handle GeoTiff, img and other image formats commonly used in remote sensing.

According to the actual demand, the user can choose `paddlers.transforms.decode_image()` or `paddlers.transforms.DecodeImg` to read data. `DecodeImg` is one of[Data transformation operators](#Data transformation operators), can be combined with other operators. `decode_image` is the encapsulation of `DecodeImg` operator, which is convenient use in the way of function calls.

The argument lish of `decode_image()` function is as follows:

|Parameter Name|Type|Parameter Description|Default Value|
|-------|----|--------|-----|
|`im_path`|`str`|Path of input image.||
|`to_rgb`|`bool`|If `True`, the conversion of BGR to RGB format is performed. This parameter is not used and may be removed in the future. Do not use it if possible.|`True`|
|`to_uint8`|`bool`|If `True`, the image data read is quantized and converted to uint8.|`True`|
|`decode_bgr`|`bool`|If `True`, automatically parses non-geo format images (such as jpeg images) into BGR format.|`True`|
|`decode_sar`|`bool`|If `True`, single-channel geo-format images (such as GeoTiff images) are automatically parsed as SAR images.|`True`|
|`read_geo_info`|`bool`|If `True`, the geographic information is read from the image.|`False`|
|`use_stretch`|`bool`|Whether to stretch the image brightness by 2% linear. Take effect only if `to_uint8` is `True`.|`False`|
|`read_raw`|`bool`|If `True`, it is equivalent to specifying `to_rgb` as `True` and `to_uint8` as `False`, and this parameter has a higher priority than the above.|`False`|

The return format is as follows:

- If `read_geo_info` is `False`, the image data ([h, w, c] arrangement) is returned in the format of np.ndarray.
- If `read_geo_info` is `True`, a binary group is returned, in which the first element is the image data read, the second element is a dictionary, in which the key-value pair is the geographic information of the image, such as geographic transformation information, geographic projection information, etc.

## Data Transformation Operator

In PaddleRS a series of classes are defined that, when instantiated, perform certain data preprocessing or data enhancement operations by calling the `__call__` method. PaddleRS calls these classes data preprocessing/data enhancement operators, and collectively **Data Transform Operators**. All data transformation operators inherit from the parent class[`Transform`](https://github.com/PaddlePaddle/PaddleRS/blob/develop/paddlers/transforms/operators.py).

### `Transform`

The `__call__` method of the `Transform` object takes a unique argument `sample`. `sample` must be a dictionary or a sequence of dictionaries. When `sample` is a sequence, perform data transformations for each dictionary in `sample` and return the results sequentially stored in a Python build-in list; When `sample` is a dictionary, the `Transform` object extracts input from some of its key-value pairs (these keys are called "input keys"), performs the transformation, and writes the results as key-value pairs into `sample`(these keys are called "output keys"). It should be noted that many of the `Transform` objects in PaddleRS currently have a carbon copy behavior, that is, an intersection between the input key and the output key. The common key names in `sample` and their meanings are as follows:

|Key Name|Description|
|----|----|
|`'image'`|Image path or data. For change detection task, it refers to the first phase image data.|
|`'image2'`|Second phase image data in change detection task.|
|`'image_t1'`|First phase image path in change detection task.|
|`'image_t2'`|Second phase image path in change detection task.|
|`'mask'`|Truth label path or data in image segmentation/change detection task.|
|`'aux_masks'`|Auxiliary label path or data in image segmentation/change detection tasks.|
|`'gt_bbox'`|Detection box labeling data in object detection task.|
|`'gt_poly'`|Polygon labeling data in object detection task.|
|`'target'`|Target image path or data in image restoration task.|

### Combined Data Transformation Operator

Use `paddlers.transforms.Compose` to combine a set of data transformation operators. `Compose` receives a list input when constructed. When you call `Compose`, it serially execute each data transform operator in the list. The following is an example:

```python
# Compose a variety of transformations using Compose.
# The transformations contained in Compose will be executed sequentially in sequence
train_transforms = T.Compose([
    # Read Image
    T.DecodeImg(),
    # Scale the image to 512x512
    T.Resize(target_size=512),
    # Perform a random horizontal flip with a 50% probability
    T.RandomHorizontalFlip(prob=0.5),
    # Normalized data to [-1,1]
    T.Normalize(
        mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    # Select and organize the information that needs to be used later
    T.ArrangeSegmenter('train')
])
```

Generally, in the list of `Compose` object accepted data transform operators, the first element is `paddlers.Transforms.DecodeImg` object, used to read image data; The last element is [`Arrange` Operator](https://github.com/PaddlePaddle/PaddleRS/blob/develop/paddlers/transforms/operators.py, used to extract and arrange information from the `sample` dictionary.

For the verification set of image segmentation task and change detection task, the `ReloadMask` operator can be inserted before the `Arrange` operator to reload the GT label. The following is an example:

```python
eval_transforms = T.Compose([
    T.DecodeImg(),
    T.Resize(target_size=512),
    T.Normalize(
        mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    # Reload label
    T.ReloadMask(),
    T.ArrangeSegmenter('eval')
])
```
