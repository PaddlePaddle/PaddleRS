简体中文 | [English](data_en.md)

# 数据相关API说明

## 数据集

在PaddleRS中，所有数据集均继承自父类[`BaseDataset`](https://github.com/PaddlePaddle/PaddleRS/blob/develop/paddlers/datasets/base.py)。

### 变化检测数据集`CDDataset`

`CDDataset`定义在：https://github.com/PaddlePaddle/PaddleRS/blob/develop/paddlers/datasets/cd_dataset.py

其初始化参数列表如下：

|参数名称|类型|参数说明|默认值|
|-------|----|--------|-----|
|`data_dir`|`str`|数据集存放目录。||
|`file_list`|`str`|file list路径。file list是一个文本文件，其中每一行包含一个样本的路径信息。`CDDataset`对file list的具体要求请参见下文。||
|`transforms`|`paddlers.transforms.Compose` \| `list`|对输入数据应用的数据变换算子。||
|`label_list`|`str` \| `None`|label list文件。label list是一个文本文件，其中每一行包含一个类别的名称。|`None`|
|`num_workers`|`int` \| `str`|加载数据时使用的辅助进程数。若设置为`'auto'`，则按照如下规则确定使用进程数：当CPU核心数大于16时，使用8个数据读取辅助进程；否则，使用CPU核心数一半数量的辅助进程。|`'auto'`|
|`shuffle`|`bool`|是否随机打乱数据集中的样本。|`False`|
|`with_seg_labels`|`bool`|当数据集中包含每个时相的分割标签时，请指定此选项为`True`。|`False`|
|`binarize_labels`|`bool`|若为`True`，则在所有数据变换算子处理完毕后对变化标签（和分割标签）进行二值化操作。例如，将取值为{0, 255}的标签二值化到{0, 1}。|`False`|

`CDDataset`对file list的要求如下：

- 当`with_seg_labels`为`False`时，file list中的每一行应该包含3个以空格分隔的项，依次表示第一时相影像相对`data_dir`的路径、第二时相影像相对`data_dir`的路径以及变化标签相对`data_dir`的路径。
- 当`with_seg_labels`为`True`时，file list中的每一行应该包含5个以空格分隔的项，其中前3项的表示含义与`with_seg_labels`为`False`时相同，后2项依次表示第一时相和第二时相影像对应的分割标签相对`data_dir`的路径。

### 场景分类数据集`ClasDataset`

`ClasDataset`定义在：https://github.com/PaddlePaddle/PaddleRS/blob/develop/paddlers/datasets/clas_dataset.py

其初始化参数列表如下：

|参数名称|类型|参数说明|默认值|
|-------|----|--------|-----|
|`data_dir`|`str`|数据集存放目录。||
|`file_list`|`str`|file list路径。file list是一个文本文件，其中每一行包含一个样本的路径信息。`ClasDataset`对file list的具体要求请参见下文。||
|`transforms`|`paddlers.transforms.Compose` \| `list`|对输入数据应用的数据变换算子。||
|`label_list`|`str` \| `None`|label list文件。label list是一个文本文件，其中每一行包含一个类别的名称。|`None`|
|`num_workers`|`int` \| `str`|加载数据时使用的辅助进程数。若设置为`'auto'`，则按照如下规则确定使用进程数：当CPU核心数大于16时，使用8个数据读取辅助进程；否则，使用CPU核心数一半数量的辅助进程。|`'auto'`|
|`shuffle`|`bool`|是否随机打乱数据集中的样本。|`False`|

`ClasDataset`对file list的要求如下：

- file list中的每一行应该包含2个以空格分隔的项，依次表示输入影像相对`data_dir`的路径以及影像的类别ID（可解析为整型值）。

### COCO格式目标检测数据集`COCODetDataset`

`COCODetDataset`定义在：https://github.com/PaddlePaddle/PaddleRS/blob/develop/paddlers/datasets/coco.py

其初始化参数列表如下：

|参数名称|类型|参数说明|默认值|
|-------|----|--------|-----|
|`data_dir`|`str`|数据集存放目录。||
|`image_dir`|`str`|输入图像存放目录。||
|`anno_path`|`str`|[COCO格式](https://cocodataset.org/#home)标注文件路径。||
|`transforms`|`paddlers.transforms.Compose` \| `list`|对输入数据应用的数据变换算子。||
|`label_list`|`str` \| `None`|label list文件。label list是一个文本文件，其中每一行包含一个类别的名称。|`None`|
|`num_workers`|`int` \| `str`|加载数据时使用的辅助进程数。若设置为`'auto'`，则按照如下规则确定使用进程数：当CPU核心数大于16时，使用8个数据读取辅助进程；否则，使用CPU核心数一半数量的辅助进程。|`'auto'`|
|`shuffle`|`bool`|是否随机打乱数据集中的样本。|`False`|
|`allow_empty`|`bool`|是否向数据集中添加负样本。|`False`|
|`empty_ratio`|`float`|负样本占比，仅当`allow_empty`为`True`时生效。若`empty_ratio`为负值或大于等于1，则保留所有生成的负样本。|`1.0`|
|`batch_transforms`|`paddlers.transforms.BatchCompose` \| `list`|对输入数据应用的批数据变换算子。||

### VOC格式目标检测数据集`VOCDetDataset`

`VOCDetDataset`定义在：https://github.com/PaddlePaddle/PaddleRS/blob/develop/paddlers/datasets/voc.py

其初始化参数列表如下：

|参数名称|类型|参数说明|默认值|
|-------|----|--------|-----|
|`data_dir`|`str`|数据集存放目录。||
|`file_list`|`str`|file list路径。file list是一个文本文件，其中每一行包含一个样本的路径信息。`VOCDetDataset`对file list的具体要求请参见下文。||
|`transforms`|`paddlers.transforms.Compose` \| `list`|对输入数据应用的数据变换算子。||
|`label_list`|`str` \| `None`|label list文件。label list是一个文本文件，其中每一行包含一个类别的名称。|`None`|
|`num_workers`|`int` \| `str`|加载数据时使用的辅助进程数。若设置为`'auto'`，则按照如下规则确定使用进程数：当CPU核心数大于16时，使用8个数据读取辅助进程；否则，使用CPU核心数一半数量的辅助进程。|`'auto'`|
|`shuffle`|`bool`|是否随机打乱数据集中的样本。|`False`|
|`allow_empty`|`bool`|是否向数据集中添加负样本。|`False`|
|`empty_ratio`|`float`|负样本占比，仅当`allow_empty`为`True`时生效。若`empty_ratio`为负值或大于等于1，则保留所有生成的负样本。|`1.0`|
|`batch_transforms`|`paddlers.transforms.BatchCompose` \| `list`|对输入数据应用的批数据变换算子。||

`VOCDetDataset`对file list的要求如下：

- file list中的每一行应该包含2个以空格分隔的项，依次表示输入影像相对`data_dir`的路径以及[Pascal VOC格式](http://host.robots.ox.ac.uk/pascal/VOC/)标注文件相对`data_dir`的路径。

### 图像复原数据集`ResDataset`

`ResDataset`定义在：https://github.com/PaddlePaddle/PaddleRS/blob/develop/paddlers/datasets/res_dataset.py

其初始化参数列表如下：

|参数名称|类型|参数说明|默认值|
|-------|----|--------|-----|
|`data_dir`|`str`|数据集存放目录。||
|`file_list`|`str`|file list路径。file list是一个文本文件，其中每一行包含一个样本的路径信息。`ResDataset`对file list的具体要求请参见下文。||
|`transforms`|`paddlers.transforms.Compose` \| `list`|对输入数据应用的数据变换算子。||
|`num_workers`|`int` \| `str`|加载数据时使用的辅助进程数。若设置为`'auto'`，则按照如下规则确定使用进程数：当CPU核心数大于16时，使用8个数据读取辅助进程；否则，使用CPU核心数一半数量的辅助进程。|`'auto'`|
|`shuffle`|`bool`|是否随机打乱数据集中的样本。|`False`|
|`sr_factor`|`int` \| `None`|对于超分辨率重建任务，指定为超分辨率倍数；对于其它任务，指定为`None`。|`None`|

`ResDataset`对file list的要求如下：

- file list中的每一行应该包含2个以空格分隔的项，依次表示输入影像（例如超分辨率重建任务中的低分辨率影像）相对`data_dir`的路径以及目标影像（例如超分辨率重建任务中的高分辨率影像）相对`data_dir`的路径。

### 图像分割数据集`SegDataset`

`SegDataset`定义在：https://github.com/PaddlePaddle/PaddleRS/blob/develop/paddlers/datasets/seg_dataset.py

其初始化参数列表如下：

|参数名称|类型|参数说明|默认值|
|-------|----|--------|-----|
|`data_dir`|`str`|数据集存放目录。||
|`file_list`|`str`|file list路径。file list是一个文本文件，其中每一行包含一个样本的路径信息。`SegDataset`对file list的具体要求请参见下文。||
|`transforms`|`paddlers.transforms.Compose` \| `list`|对输入数据应用的数据变换算子。||
|`label_list`|`str` \| `None`|label list文件。label list是一个文本文件，其中每一行包含一个类别的名称。|`None`|
|`num_workers`|`int` \| `str`|加载数据时使用的辅助进程数。若设置为`'auto'`，则按照如下规则确定使用进程数：当CPU核心数大于16时，使用8个数据读取辅助进程；否则，使用CPU核心数一半数量的辅助进程。|`'auto'`|
|`shuffle`|`bool`|是否随机打乱数据集中的样本。|`False`|

`SegDataset`对file list的要求如下：

- file list中的每一行应该包含2个以空格分隔的项，依次表示输入影像相对`data_dir`的路径以及分割标签相对`data_dir`的路径。

## 数据读取API

遥感影像的来源多样，数据格式十分繁杂。PaddleRS为不同类型、不同格式的遥感影像提供了统一的读取接口。目前，PaddleRS支持.png、.jpg、.bmp、.npy等常见文件格式的读取，也支持处理遥感领域常用的GeoTiff、img等影像格式。

根据实际需要，用户可以选择`paddlers.transforms.decode_image()`或`paddlers.transforms.DecodeImg`进行数据读取。`DecodeImg`是[数据变换算子](#数据变换算子)之一，可以与其它算子组合使用。`decode_image`是对`DecodeImg`算子的封装，方便用户以函数调用的方式使用。

`decode_image()`函数的参数列表如下：

|参数名称|类型|参数说明|默认值|
|-------|----|--------|-----|
|`im_path`|`str`|输入图像路径。||
|`to_rgb`|`bool`|若为`True`，则执行BGR到RGB格式的转换。该参数已废弃，在将来可能被移除，请尽可能避免使用。|`True`|
|`to_uint8`|`bool`|若为`True`，则将读取的影像数据量化并转换为uint8类型。|`True`|
|`decode_bgr`|`bool`|若为`True`，则自动将非地学格式影像（如jpeg影像）解析为BGR格式。|`True`|
|`decode_sar`|`bool`|若为`True`，则自动将单通道的地学格式影像（如GeoTiff影像）作为SAR影像解析。|`True`|
|`read_geo_info`|`bool`|若为`True`，则从影像中读取地理信息。|`False`|
|`use_stretch`|`bool`|是否对影像亮度进行2%线性拉伸。仅当`to_uint8`为`True`时有效。|`False`|
|`read_raw`|`bool`|若为`True`，等价于指定`to_rgb`为`True`而`to_uint8`为`False`，且该参数的优先级高于上述参数。|`False`|

返回格式如下：

- 若`read_geo_info`为`False`，则以`numpy.ndarray`形式返回读取的影像数据（[h, w, c]排布）；
- 若`read_geo_info`为`True`，则返回一个二元组，其中第一个元素为读取的影像数据，第二个元素为一个字典，其中的键值对为影像的地理信息，如地理变换信息、地理投影信息等。

## 数据变换算子

在PaddleRS中定义了一系列类，这些类在实例化之后，可通过调用`__call__`方法执行某种特定的数据预处理或数据增强操作。PaddleRS将这些类称为数据预处理/数据增强算子，并统称为**数据变换算子**。所有数据变换算子均继承自父类[`Transform`](https://github.com/PaddlePaddle/PaddleRS/blob/develop/paddlers/transforms/operators.py)。

### `Transform`

`Transform`对象的`__call__`方法接受唯一的参数`sample`。`sample`必须为字典或字典构成的序列。当`sample`是序列时，为`sample`中的每个字典执行数据变换操作，并将变换结果依次存储在一个Python built-in list中返回；当`sample`是字典时，`Transform`对象根据其中的一些键值对提取输入（这些键称为“输入键”），执行变换后，将结果以键值对的形式写入`sample`中（这些键称为“输出键”）。需要注意的是，目前PaddleRS中许多`Transform`对象都存在复写行为，即，输入键与输出键之间存在交集。`sample`中常见的键名及其表示的含义如下表：

|键名|说明|
|----|----|
|`'image'`|影像路径或数据。对于变化检测任务，指第一时相影像数据。|
|`'image2'`|变化检测任务中第二时相影像数据。|
|`'image_t1'`|变化检测任务中第一时相影像路径。|
|`'image_t2'`|变化检测任务中第二时相影像路径。|
|`'mask'`|图像分割/变化检测任务中的真值标签路径或数据。|
|`'aux_masks'`|图像分割/变化检测任务中的辅助标签路径或数据。|
|`'gt_bbox'`|目标检测任务中的检测框标注数据。|
|`'gt_poly'`|目标检测任务中的多边形标注数据。|
|`'target'`|图像复原中的目标影像路径或数据。|

### 构造数据变换算子

请参考[此文档](../intro/transforms_cons_params_cn.md)。

### 组合数据变换算子

使用`paddlers.transforms.Compose`对一组数据变换算子进行组合。`Compose`对象在构造时接受一个列表输入。在调用`Compose`对象时，相当于串行执行列表中的每一个数据变换算子。示例如下：

```python
# 使用Compose组合多种变换方式。Compose中包含的变换将按顺序串行执行
train_transforms = T.Compose([
    # 将影像缩放到512x512大小
    T.Resize(target_size=512),
    # 以50%的概率实施随机水平翻转
    T.RandomHorizontalFlip(prob=0.5),
    # 将数据归一化到[-1,1]
    T.Normalize(
        mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
```

对于图像分割任务和变化检测任务的验证集而言，可使用`ReloadMask`算子重新加载真值标签。示例如下：

```python
eval_transforms = T.Compose([
    T.Resize(target_size=512),
    T.Normalize(
        mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    # 重新加载标签
    T.ReloadMask()
])
```
