# Data related API description

## Dataset

In PaddleRS, all datasets inherit from the parent class [`BaseDataset`](https://github.com/PaddlePaddle/PaddleRS/blob/develop/paddlers/datasets/base.py)。

### Change detection datasets `CDDataset`

`CDDataset` is defined in：https://github.com/PaddlePaddle/PaddleRS/blob/develop/paddlers/datasets/cd_dataset.py

The initialization parameter list is as follows：

|Parameter names|Class|Parameter description|Default|
|-------|----|--------|-----|
|`data_dir`|`str`|Storage path of dataset||
|`file_list`|`str`|file list path .file list is a text file，which contains the path information for a sample in each line. The definite requirements of 'CDDataset' on the file list are listed below.||
|`transforms`|`paddlers.transforms.Compose`|Data transformation operator applied to input data.||
|`label_list`|`str` \| `None`|label list  file. label list is a text file，which contains the path information for a sample in each line。|`None`|
|`num_workers`|`int` \| `str`|The number of worker processes used when loading data.If it was set at `'auto'`，The number of processes in use is determined according to the following rules: When the number of CPU cores is greater than 16, 8 data read helper processes are used. Otherwise, use CPU cores to count half as many worker processes.|`'auto'`|
|`shuffle`|`bool`|Whether to randomly shuffle the samples in the data set.|`False`|
|`with_seg_labels`|`bool`|Specify this option as' True 'when the dataset contains split labels for each time phase.|`False`|
|`binarize_labels`|`bool`|If 'True', then binarize the change labels (and split labels) after all data transformation operators except 'Arrange' have been processed. For example, the label {0, 255} is binarized to {0, 1}.|`False`|

`CDDataset` the requirements for a file list are as follows：

- When 'with_seg_labels' is' False', each line of the file list should contain three items separated by Spaces. Denote the path of the first phase image relative to 'data_dir', the path of the second phase image relative to 'data_dir', and the path of the change label relative to 'data_dir'.
- When 'with_seg_labels' is' True', each line in the file list should contain five whitespace-separated items, the first three of which have the same meaning as when 'with_seg_labels' is' False'. The last two terms represent the path of the segmentation label corresponding to the first and second temporal images to 'data_dir' in turn.

### Scene classification dataset `ClasDataset`


`ClasDataset` is defined in: https://github.com/PaddlePaddle/PaddleRS/blob/develop/paddlers/

The list of initialized parameters is as follows:

|Parameter names|Type|Parameter Description|Default|
|-------|----|--------|-----|
|`data_dir`|`str`|The directory where the dataset stored.||
|`file_list`|`str`|file list path. The file list is a text file where each line contains the path information for one sample. See the 'ClasDataset' file list requirements below.||
|`transforms`|`paddlers.transforms.Compose`|A data transformation operator applied to input data.||
|`label_list`|`str` \| `None`|label list file. A label list is a text file where each line contains the name of a category.|`None`|
|`num_workers`|`int` \| `str`|Number of helper processes used to load the data. If set to "auto' ', the number of processes used is determined according to the following rules: when the number of CPU cores is greater than 16, 8 data reading helper processes are used; Otherwise, half as many helper processes as CPU cores are used.|`'auto'`|
|`shuffle`|`bool`|Whether to randomly shuffle the samples in the dataset.|`False`|

The 'ClasDataset' file list is as follows:

- Each line in the file list should contain 2 whitespace-separated items representing the path of the input image to the 'data_dir' and the category ID of the image (which can be resolved as an integer).

### COCO format object detection dataset 'COCODetDataset'

`COCODetDataset` defined in: https://github.com/PaddlePaddle/PaddleRS/blob/develop/paddlers/datasets/coco.py

The list of initialized parameters is as follows:

|Parameter names|Types|Parameter Descirption|Default|
|-------|----|--------|-----|
|`data_dir`|`str`|Data set storage directory.||
|`image_dir`|`str`|Enter the directory where the image is stored.||
|`ann_path`|`str`|[COCO format](https://cocodataset.org/#home) specifies the file path.||
|`transforms`|`paddlers.transforms.Compose`|A data transformation operator applied to input data.||
|`label_list`|`str` \| `None`|label list file. A label list is a text file where each line contains the name of a category.|`None`|
|`num_workers`|`int` \| `str`|Number of helper processes used to load the data. If set to "auto' ', the number of processes used is determined according to the following rules: when the number of CPU cores is greater than 16, 8 data reading helper processes are used; Otherwise, half as many helper processes as CPU cores are used.|`'auto'`|
|`shuffle`|`bool`|Whether to randomly shuffle the samples in the data set.|`False`|
|`allow_empty`|`bool`|Whether to add negative samples to the dataset.|`False`|
|`empty_ratio`|`float`|Negative percentage, only if 'allow_empty' is' True '. If 'empty_ratio' is negative or greater than or equal to 1, then all generated negative samples are kept.|`1.0`|

### VOC format object detection dataset 'VOCDetDataset'

`VOCDetDataset`defined in：https://github.com/PaddlePaddle/PaddleRS/blob/develop/paddlers/datasets/voc.py

The list of initialized parameters is as follows:

|Parameter names|Types|Parameter Descirption|Default|
|-------|----|--------|-----|
|`data_dir`|`str`|The directory where the dataset is stored.||
|`file_list`|`str`|file list path. The file list is a text file where each line contains the path information for one sample. See the file list requirements for 'VOCDetDataset' below.||
|`transforms`|`paddlers.transforms.Compose`|A data transformation operator applied to input data.||
|`label_list`|`str` \| ` None ` | label list file. A label list is a text file where each line contains the name of a category.|`None`|
|`num_workers`|`int` \| `str`|Number of helper processes used to load the data. If set to "auto' ', the number of processes used is determined according to the following rules: when the number of CPU cores is greater than 16, 8 data reading helper processes are used; Otherwise, half as many helper processes as CPU cores are used.|`'auto'`|
|`shuffle`|`bool`|Whether to randomly shuffle the samples in the dataset.|`False`|
|`allow_empty`|`bool`|Whether to add negative samples to the dataset.|`False`|
|`empty_ratio`|`float`|Negative sample ratio, effective only if 'allow_empty' is' True '. If 'empty_ratio' is negative or greater than or equal to 1, all negative samples generated are retained.|`1.0`|

The  `VOCDetDataset'requires the following for a file list：

- file list in each line should contain two separated by Spaces, in turn, said the input image is relatively ` data_dir ` path and [Pascal VOC format] (http://host.robots.ox.ac.uk/pascal/VOC/) tagging files relative ` data_dir ` paths.

### Image Restoration dataset`ResDataset`

`ResDataset` defined in the：https://github.com/PaddlePaddle/PaddleRS/blob/develop/paddlers/datasets/res_dataset.py

The list of initialization parameters is as follows：

|Parameter names|Types|Parameter Description|Default|
|-------|----|--------|-----|
|`data_dir`|`str`|The directory where the dataset stored.||
|`file_list`|`str`|file list path. The file list is a text file where each line contains path information for a sample. See the file list requirements for 'ResDataset' below.||
|`transforms`|`paddlers.transforms.Compose`|Data transformation operator applied to input data.||
|`num_workers`|`int` \| `str`|Number of helper processes used to load the data. If set to "auto' ', the number of processes used is determined according to the following rules: when the number of CPU cores is greater than 16, 8 data reading helper processes are used; Otherwise, half as many helper processes as CPU cores are used.|`'auto'`|
|`shuffle`|`bool`|Whether to randomly shuffle the samples in the dataset.|`False`|
|`sr_factor`|`int` \| `None`|For the super-resolution reconstruction task, is the super-resolution multiple; For other tasks, specify 'None'.|`None`|

The 'ResDataset' requires the following for the file list：

- Each line of the file list should contain 2 whitespace separated items, which represent the path of the input image (e.g. low-resolution image in the super-resolution reconstruction task) relative to 'data_dir' and the path of the target image (e.g. high-resolution image in the super-resolution reconstruction task) relative to 'data_dir'.

### Image segmentation dataset `SegDataset`

`SegDataset` is defined in ：https://github.com/PaddlePaddle/PaddleRS/blob/develop/paddlers/datasets/seg_dataset.py

The list of initialization parameters is as follows：

|Parameter names|Types|Parameter Description|Default|
|-------|----|--------|-----|
|`data_dir`|`str`|The directory where the dataset stored.||
|`file_list`|`str`|file list path. The file list is a text file where each line contains the path information for a sample.`SegDataset` For details about the requirements on the file list, see the following section.||
|`transforms`|`paddlers.transforms.Compose`|Data transformation operator applied to input data.||
|`label_list`|`str` \| `None`|label list file. label list is a text file where each line contains the name of a category.|`None`|
|`num_workers`|`int` \| `str`|Number of helper processes used to load the data. If set to "auto' ', the number of processes used is determined according to the following rules: when the number of CPU cores is greater than 16, 8 data reading helper processes are used; Otherwise, half as many helper processes as CPU cores are used.|`'auto'`|
|`shuffle`|`bool`|Whether to randomly shuffle the samples in the dataset.|`False`|

The 'SegDataset' requires the following for the file list：

- Each line in the file list should contain 2 whitespace-separated items, representing the path of the input image relative to 'data_dir' and the path of the split label relative to 'data_dir'.

## Data fetching API

The sources of remote sensing images are various, and the data formats are very complex. PaddleRS provides a unified reading interface for different types and formats of remote sensing images. Currently, PaddleRS supports reading common file formats such as.png,.jpg,.bmp,.npy, and also supports processing GeoTiff, img and other image formats commonly used in the field of remote sensing.

According to the actual needs, the user can choose 'paddlers.transforms.decode_image()' or `paddlers.transforms.DecodeImg` to read the data. 'DecodeImg' is one of [data transform operators](# data transform operators), can be used in combination with other operators. 'decode_image' is the encapsulation of 'DecodeImg' operator, which is convenient for users to use in the way of function calls.
`decode_image()` argument list is as follows：

|Parameter names|Types|Parameter Description|Default|
|-------|----|--------|-----|
|`im_path`|`str`|Input image path.||
|`to_rgb`|`bool`|If`True`，BGR to RGB format conversion is performed. This parameter is deprecated and may be removed in the future, so avoid using it if possible.|`True`|
|`to_uint8`|`bool`|If`True`，The read image data is quantized and converted to uint8Types.|`True`|
|`decode_bgr`|`bool`|If`True`，Non-geotechnical image format (such as jpeg image) is automatically parsed into BGR format.|`True`|
|`decode_sar`|`bool`|If`True`，Then single-channel geoscience format images (such as GeoTiff images) are automatically parsed as SAR images.|`True`|
|`read_geo_info`|`bool`|If`True`，Then reading geographic information from images.|`False`|
|`use_stretch`|`bool`|Whether the image brightness should be linearly stretched by 2%. Only if 'to_uint8' is 'True'.|`False`|
|`read_raw`|`bool`|If`True`，This is equivalent to specifying 'to_rgb' as' True 'and' to_uint8 'as' False', and this parameter has a higher priority than the above.|`False`|

The return format is as follows:

- If 'read_geo_info' is' False ', then the image data ([h, w, c] arrangement) is returned in np-ndarray form;
- If 'read_geo_info' is' True ', a binary group is returned, where the first element is the image data read, and the second element is a dictionary, in which the key-value pairs are the geographic information of the image, such as geographic transformation information, geographic projection information, etc.

## Data transformation operators

PaddleRS defines a series of classes that, when instantiated, can perform a specific data preprocessing or data augmentation operation by calling the '__call__' method. PaddleRS refer to these classes as data preprocessing/data augmentation operators and collectively refer to them as ** data transformation operators **.All data transformation operators are inherited from the parent class[`Transform`](https://github.com/PaddlePaddle/PaddleRS/blob/develop/paddlers/transforms/operators.py).

### `Transform`

The '__call__' method of the 'Transform' object accepts a single argument, 'sample'.'sample' must be a dictionary or a sequence of dictionaries. When 'sample' is a sequence, a data transform operation is performed for each dictionary in 'sample'，
The transformation results are stored in a Python built-in list and will be returned；When 'sample' is a dictionary, the 'Transform' object extracts the input based on some key-value pairs (these keys are called "input keys"), performs the transformation, and writes the result to 'sample' as key-value pairs (these keys are called "output keys").It should be noted that many of the Transform objects in PaddleRS currently have a carbon copy behavior, that is, an intersection between the input key and the output key. The common key names in 'sample' and their meanings are as follows:

|Key name|Explanation|
|----|----|
|`'image'`|Image path or data. For the change detection task, it refers to the first phase image data.|
|`'image2'`|The second temporal image data in the change detection task.|
|`'image_t1'`|The first temporal image path in the change detection task.|
|`'image_t2'`|The second temporal image path in the change detection task.|
|`'mask'`|Truth label path or data in image segmentation/change detection task.|
|`'aux_masks'`|Auxiliary tag path or data in image segmentation/change detection task.|
|`'gt_bbox'`|Detection box annotation data in the object detection task.|
|`'gt_poly'`|Polygon labeling data in target detection task.|
|`'target'`|Target image path or data in image restoration.|

### Combined data transformation operators

Combine a set of data transformation operators using 'paddlers.transforms.Compose'. The 'Compose' object receives a list input when constructed. When you call 'Compose' object, you serially execute each data transform operator in the list. The following is an example:

```python
# Compose a variety of transformations using 'Compose'. The transformations contained in Compose will be executed sequentially in sequence.
train_transforms = T.Compose([
    # Reading images
    T.DecodeImg(),
    # Resize the image to 512x512
    T.Resize(target_size=512),
    # Random horizontal flips are implemented with 50% probability
    T.RandomHorizontalFlip(prob=0.5),
    # Normalize the data to [-1,1]
    T.Normalize(
        mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    # Select and organize the information that needs to be used later.
    T.ArrangeSegmenter('train')
])
```

In general,`Compose` object accept data transform operator in the list, the first element for `paddlers.transforms.DecodeImg` object, used to read image data;The last element for [` Arrange ` operator] (https://github.com/PaddlePaddle/PaddleRS/blob/develop/paddlers/transforms/operators.py), Used to extract and arrange information from the 'sample' dictionary.

For the validation set of the image segmentation task and change detection task, the 'ReloadMask' operator can be inserted before the 'Arrange' operator to reload the truth labels. An example is as follows:

```python
eval_transforms = T.Compose([
    T.DecodeImg(),
    T.Resize(target_size=512),
    T.Normalize(
        mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    # Reloading the label
    T.ReloadMask(),
    T.ArrangeSegmenter('eval')
])
```
