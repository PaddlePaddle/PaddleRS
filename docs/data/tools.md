# 工具箱

工具箱位于`tools`文件夹下，目前有如下工具：

- `coco2mask`：用于将geojson格式的分割标注标签转换为png格式。
- `mask2shp`：用于对推理得到的png提取shapefile。
- `mask2geojson`：用于对推理得到的png提取geojson。
- `geojson2mask`：用于从geojson和原图中提取mask作为训练标签。
- `matcher`：用于在推理前匹配两个时段的影响。
- `spliter`：用于将大图数据进行分割以作为训练数据。
- `coco_tools`：用于统计处理coco类标注文件。

后期将根据DL和RS/GIS方面的前后处理需求增加更多的工具。

## 如何使用

首先需要`clone`此repo并进入到`tools`的文件夹中：

```shell
git clone https://github.com/PaddleCV-SIG/PaddleRS.git
cd PaddleRS\tools
```

### coco2mask

`coco2mask`的主要功能是将图像以及对应json格式的分割标签转换为图像与png格式的标签，结果会分别存放在`img`和`gt`两个文件夹中。相关的数据样例可以参考[中国典型城市建筑物实例数据集](https://www.scidb.cn/detail?dataSetId=806674532768153600&dataSetType=journal)。保存结果为单通道的伪彩色图像。使用代码如下：

```shell
python coco2mask.py --raw_folder xxx --save_folder xxx
```

其中：

- `raw_folder`：存放原始数据的文件夹，图像存放在`images`文件夹中，标签以`xxx.json`进行保存。
- `save_folder`：保存结果文件的文件夹，其中图像保存在`img`中，png格式的标签保存在`gt`中。

### mask2shp

`mask2shp`的主要功能是将推理得到的png格式的分割结果转换为shapefile格式，其中还可以设置不生成多边形的索引号。使用代码如下：

```shell
python mask2shp.py --srcimg_path xxx.tif --mask_path xxx.png [--save_path output] [--ignore_index 255]
```

其中：

- `srcimg_path`：原始图像的路径，需要带有地理信息，以便为生成的shapefile提供crs等信息。
- `mask_path`：推理得到的png格式的标签的路径。
- `save_path`：保存shapefile的路径，默认为`output`。
- `ignore_index`：忽略生成shp的索引，如背景等，默认为255。

### mask2geojson

`mask2geojson`的主要功能是将推理得到的png格式的分割结果转换为geojson格式。使用代码如下：

```shell
python mask2geojson.py --mask_path xxx.tif --save_path xxx.json [--epsilon 0]
```

其中：

- `mask_path`：推理得到的png格式的标签的路径。
- `save_path`：保存geojson的路径。
- `epsilon`：opencv的简化参数，默认为0。

### geojson2mask

`geojson2mask`的主要功能是从原图和geojson文件中提取mask图像。使用代码如下：

```shell
python  geojson2mask.py --image_path xxx.tif --geojson_path xxx.json
```

其中：

- `image_path`：原图像的路径。
- `geojson_path`：geojson的路径。

### matcher

` matcher`的主要功能是在进行变化检测的推理前，匹配两期影像的位置，并将转换后的`im2`图像保存在原地址下，命名为`im2_M.tif`。使用代码如下：

```shell
python matcher.py --im1_path xxx.tif --im2_path xxx.xxx [--im1_bands 1 2 3] [--im2_bands 1 2 3]
```

其中：

- `im1_path`：时段一的图像路径，该图像需要存在地理信息，且以该图像为基准图像。
- `im2_path`：时段二的图像路径，该图像可以为非遥感格式的图像，该图像为带匹配图像。
- `im1_bands`：时段一图像所用于配准的波段，为RGB或单通道，默认为[1, 2, 3]。
- `im2_bands`：时段二图像所用于配准的波段，为RGB或单通道，默认为[1, 2, 3]。

### spliter

`spliter`的主要功能是在划分大的遥感图像为图像块，便于进行训练。使用代码如下：

```shell
python spliter.py --image_path xxx.tif [--mask_path None] [--block_size 512] [--save_folder output]
```

其中：

- `image_path`：需要切分的图像的路径。
- `mask_path`：一同切分的标签图像路径，默认没有。
- `block_size`：切分图像块大小，默认为512。
- `save_folder`：保存切分后结果的文件夹路径，默认为`output`。

### coco_tools

目前coco_tools共有6个文件，各文件及其功能如下：

* json_InfoShow:    打印json文件中各个字典的基本信息；
* json_ImgSta:      统计json文件中的图像信息，生成统计表、统计图；
* json_AnnoSta:     统计json文件中的标注信息，生成统计表、统计图；
* json_Img2Json:    统计test集图像，生成json文件；
* json_Split:       json文件拆分，划分为train set、val set
* json_Merge:       json文件合并，将多个json合并为1个json

详细使用方法与参数见[coco_tools说明](coco_tools_cn.md)
