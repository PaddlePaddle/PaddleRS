# 遥感影像处理工具集

PaddleRS在`tools`目录中提供了丰富的遥感影像处理工具，包括：

- `coco2mask.py`：用于将COCO格式的标注文件转换为png格式。
- `mask2shape.py`：用于将模型推理输出的.png格式栅格标签转换为矢量格式。
- `mask2geojson.py`：用于将模型推理输出的.png格式栅格标签转换为GeoJSON格式。
- `match.py`：用于实现两幅影像的配准。
- `split.py`：用于对大幅面影像数据进行切片。
- `coco_tools/`：COCO工具合集，用于统计处理COCO格式标注文件。
- `prepare_dataset/`：数据集预处理脚本合集。

## 使用说明

首先请确保您已将PaddleRS下载到本地。进入`tools`目录：

```shell
cd tools
```

### coco2mask

`coco2mask.py`的主要功能是将图像以及对应的COCO格式的分割标签转换为图像与.png格式的标签，结果会分别存放在`img`和`gt`两个目录中。相关的数据样例可以参考[中国典型城市建筑物实例数据集](https://www.scidb.cn/detail?dataSetId=806674532768153600&dataSetType=journal)。对于mask，保存结果为单通道的伪彩色图像。使用方式如下：

```shell
python coco2mask.py --raw_dir {输入目录路径} --save_dir {输出目录路径}
```

其中：

- `raw_dir`：存放原始数据的目录，其中图像存放在`images`子目录中，标签以`xxx.json`格式保存。
- `save_dir`：保存输出结果的目录，其中图像保存在`img`子目录中，.png格式的标签保存在`gt`子目录中。

### mask2shape

`mask2shape.py`的主要功能是将.png格式的分割结果转换为shapefile格式（矢量图）。使用方式如下：

```shell
python mask2shape.py --srcimg_path {带有地理信息的原始影像路径} --mask_path {输入分割标签路径} [--save_path {输出矢量图路径}] [--ignore_index {需要忽略的索引值}]
```

其中：

- `srcimg_path`：原始影像路径，需要带有地理坐标信息，以便为生成的shapefile提供crs等信息。
- `mask_path`：模型推理得到的.png格式的分割结果。
- `save_path`：保存shapefile的路径，默认为`output`。
- `ignore_index`：需要在shapefile中忽略的索引值（例如分割任务中的背景类），默认为255。

### mask2geojson

`mask2geojson.py`的主要功能是将.png格式的分割结果转换为GeoJSON格式。使用方式如下：

```shell
python mask2geojson.py --mask_path {输入分割标签路径} --save_path {输出路径}
```

其中：

- `mask_path`：模型推理得到的.png格式的分割结果。
- `save_path`：保存GeoJSON文件的路径。

### match

`match.py`的主要功能是在对两个时相的遥感影像进行空间配准。使用方式如下：

```shell
python match.py --im1_path [时相1影像路径] --im2_path [时相2影像路径] --save_path [配准后时相2影像输出路径] [--im1_bands 1 2 3] [--im2_bands 1 2 3]
```

其中：

- `im1_path`：时相1影像路径。该影像必须包含地理信息，且配准过程中以该影像为基准图像。
- `im2_path`：时相2影像路径。该影像的地理信息将不被用到。配准过程中将该影像配准到时相1影像。
- `im1_bands`：时相1影像用于配准的波段，指定为三通道（分别代表R、G、B）或单通道，默认为[1, 2, 3]。
- `im2_bands`：时相2影像用于配准的波段，指定为三通道（分别代表R、G、B）或单通道，默认为[1, 2, 3]。
- `save_path`: 配准后时相2影像输出路径。

### split

`split.py`的主要功能是将大幅面遥感图像划分为图像块，这些图像块可以作为训练时的输入。使用方式如下：

```shell
python split.py --image_path {输入影像路径} [--mask_path {真值标签路径}] [--block_size {图像块尺寸}] [--save_dir {输出目录}]
```

其中：

- `image_path`：需要切分的图像的路径。
- `mask_path`：一同切分的标签图像路径，默认没有。
- `block_size`：切分图像块大小，默认为512。
- `save_folder`：保存切分后结果的文件夹路径，默认为`output`。

### coco_tools

目前`coco_tools`目录中共包含6个工具，各工具功能如下：

- `json_InfoShow.py`:    打印json文件中各个字典的基本信息；
- `json_ImgSta.py`:      统计json文件中的图像信息，生成统计表、统计图；
- `json_AnnoSta.py`:     统计json文件中的标注信息，生成统计表、统计图；
- `json_Img2Json.py`:    统计test集图像，生成json文件；
- `json_Split.py`:       将json文件中的内容划分为train set和val set；
- `json_Merge.py`:       将多个json文件合并为一个。

详细使用方法请参见[coco_tools使用说明](coco_tools.md)。

### prepare_dataset

`prepare_dataset`目录中包含一系列数据预处理脚本，主要用于预处理已下载到本地的遥感开源数据集，使其符合PaddleRS训练、验证、测试的标准。

在执行脚本前，您可以通过`--help`选项获取帮助信息。例如：

```shell
python prepare_dataset/prepare_levircd.py --help
```

以下列出了脚本中常见的命令行选项：

- `--in_dataset_dir`：下载到本地的原始数据集所在路径。示例：`--in_dataset_dir downloads/LEVIR-CD`。
- `--out_dataset_dir`：处理后的数据集存放路径。示例：`--out_dataset_dir data/levircd`。
- `--crop_size`：对于支持影像裁块的数据集，指定切分的影像块大小。示例：`--crop_size 256`。
- `--crop_stride`：对于支持影像裁块的数据集，指定切分时滑窗移动的步长。示例：`--crop_stride 256`。
- `--seed`：随机种子。可用于固定随机数生成器产生的伪随机数序列，从而得到固定的数据集划分结果。示例：`--seed 1919810`
- `--ratios`：对于支持子集随机划分的数据集，指定需要划分的各个子集的样本比例。示例：`--ratios 0.7 0.2 0.1`。

您可以在[此文档](https://github.com/PaddlePaddle/PaddleRS/blob/develop/docs/intro/data_prep.md)中查看PaddleRS提供哪些数据集的预处理脚本。
