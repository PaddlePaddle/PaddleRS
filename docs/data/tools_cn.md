简体中文 | [English](tools_en.md)

# 遥感影像处理工具集

PaddleRS在`tools`目录中提供了丰富的遥感影像处理工具，包括：

- `coco2mask.py`：用于将COCO格式的标注文件转换为PNG格式。
- `mask2shape.py`：用于将模型推理输出的PNG格式栅格标签转换为.shp矢量格式。
- `geojson2mask.py`：用于将GeoJSON格式标签转换为.tif栅格格式。
- `match.py`：用于实现两幅影像的配准。
- `split.py`：用于对大幅面影像数据进行切片。
- `coco_tools/`：COCO工具合集，用于统计处理COCO格式标注文件。
- `prepare_dataset/`：数据集预处理脚本合集。
- `extract_ms_patches.py`：从整幅遥感影像中提取多尺度影像块。
- `generate_file_lists.py`：对数据集生成file list。

## 使用说明

首先请确保您已将PaddleRS下载到本地。进入`tools`目录：

```shell
cd tools
```

### coco2mask

`coco2mask.py`的主要功能是将影像以及对应的COCO格式的分割标签转换为影像与PNG格式的标签，结果会分别存放在`img`和`gt`两个目录中。相关的数据样例可以参考[中国典型城市建筑物实例数据集](https://www.scidb.cn/detail?dataSetId=806674532768153600&dataSetType=journal)。对于mask，保存结果为单通道的伪彩色影像。使用方式如下：

```shell
python coco2mask.py --raw_dir {输入目录路径} --save_dir {输出目录路径}
```

其中：

- `--raw_dir`：存放原始数据的目录，其中影像存放在`images`子目录中，标签以`xxx.json`格式保存。
- `--save_dir`：保存输出结果的目录，其中影像保存在`img`子目录中，PNG格式的标签保存在`gt`子目录中。

### mask2shape

`mask2shape.py`的主要功能是将PNG格式的分割结果转换为shapefile格式（矢量图）。使用方式如下：

```shell
python mask2shape.py --src_img_path {带有地理信息的原始影像路径} --mask_path {输入分割标签路径} [--save_path {输出矢量图路径}] [--ignore_index {需要忽略的索引值}]
```

其中：

- `--src_img_path`：原始影像路径，需要带有地理元信息，以便为生成的shapefile提供地理投影坐标系等信息。
- `--mask_path`：模型推理得到的PNG格式的分割结果。
- `--save_path`：保存shapefile的路径，默认为`output`。
- `--ignore_index`：需要在shapefile中忽略的索引值（例如分割任务中的背景类），默认为`255`。

### geojson2mask

`geojson2mask.py`的主要功能是将GeoJSON格式的标签转换为.tif的栅格格式。使用方式如下：

```shell
python geojson2mask.py --src_img_path {带有地理信息的原始影像路径} --geojson_path {输入分割标签路径} --save_path {输出路径}
```

其中：

- `--src_img_path`：原始影像路径，需要带有地理元信息。
- `--geojson_path`：GeoJSON格式标签路径。
- `--save_path`：保存转换后的栅格文件的路径。

### match

`match.py`的主要功能是在对两个时相的遥感影像进行空间配准。使用方式如下：

```shell
python match.py --image1_path {时相1影像路径} --image2_path {时相2影像路径} --save_path {配准后时相2影像输出路径} [--image1_bands 1 2 3] [--image2_bands 1 2 3]
```

其中：

- `--image1_path`：时相1影像路径。该影像必须包含地理信息，且配准过程中以该影像为基准影像。
- `--image2_path`：时相2影像路径。该影像的地理信息将不被用到。配准过程中将该影像配准到时相1影像。
- `--image1_bands`：时相1影像用于配准的波段，指定为三通道（分别代表R、G、B）或单通道，默认为`[1, 2, 3]`。
- `--image2_bands`：时相2影像用于配准的波段，指定为三通道（分别代表R、G、B）或单通道，默认为`[1, 2, 3]`。
- `--save_path`： 配准后时相2影像输出路径。

### split

`split.py`的主要功能是将大幅面遥感影像划分为影像块，这些影像块可以作为训练时的输入。使用方式如下：

```shell
python split.py --image_path {输入影像路径} [--mask_path {真值标签路径}] [--block_size {影像块尺寸}] [--save_dir {输出目录路径}]
```

其中：

- `--image_path`：需要切分的影像的路径。
- `--mask_path`：一同切分的标签影像路径，默认为`None`。
- `--block_size`：切分影像块大小，默认为`512`。
- `--save_dir`：保存切分后结果的文件夹路径，默认为`output`。

### coco_tools

目前`coco_tools`目录中共包含6个工具，各工具功能如下：

- `json_info_show.py`：    打印JSON文件中各个字典的基本信息；
- `json_image_sta.py`：      统计JSON文件中的影像信息，生成统计表、统计图；
- `json_anno_sta.py`：     统计JSON文件中的标注信息，生成统计表、统计图；
- `json_image2json.py`：    统计test集影像，生成JSON文件；
- `json_split.py`：       将JSON文件中的内容划分为train set和val set；
- `json_merge.py`：       将多个JSON文件合并为一个。

详细使用方法请参见[coco_tools使用说明](coco_tools_cn.md)。

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

您可以在[此文档](../intro/data_prep_cn.md)中查看PaddleRS提供哪些数据集的预处理脚本。

### extract_ms_patches

`extract_ms_patches.py`的主要功能是利用四叉树从整幅遥感影像中提取不同尺度的包含感兴趣目标的影像块，提取的影像块可用作模型训练样本。使用方式如下：

```shell
python extract_ms_patches.py --image_paths {一个或多个输入影像路径} --mask_path {真值标签路径} [--save_dir {输出目录}] [--min_patch_size {最小的影像块尺寸}] [--bg_class {背景类类别编号}] [--target_class {目标类类别编号}] [--max_level {检索的最大尺度层级}] [--include_bg] [--nonzero_ratio {影像块中非零像素占比阈值}] [--visualize]
```

其中：

- `--image_paths`：源影像路径，可以指定多个路径。
- `--mask_path`：真值标签路径。
- `--save_dir`：保存切分后结果的文件夹路径，默认为`output`。
- `--min_patch_size`：提取的影像块的最小尺寸（以影像块长/宽的像素个数计），即四叉树的叶子结点在图中覆盖的最小范围，默认为`256`。
- `--bg_class`：背景类别的类别编号，默认为`0`。
- `--target_class`：目标类别的类别编号，若为`None`，则表示所有背景类别以外的类别均为目标类别，默认为`None`。
- `--max_level`：检索的最大尺度层级，若为`None`，则表示不限制层级，默认为`None`。
- `--include_bg`：若指定此选项，则也保存那些仅包含背景类别、不包含目标类别的影像块。
- `--nonzero_ratio`：指定一个阈值，对于任意一幅源影像，若影像块中非零像素占比小于此阈值，则该影像块将被舍弃。若为`None`，则表示不进行过滤。默认为`None`。
- `--visualize`：若指定此选项，则程序执行完毕后将生成图像`./vis_quadtree.png`，其中保存有四叉树中节点情况的可视化结果，一个例子如下图所示：

<div align="center">
<img src="https://user-images.githubusercontent.com/21275753/189264850-f94b3d7b-c631-47b1-9833-0800de2ccf54.png"  width = "400" />  
</div>

### generate_file_lists

`generate_file_lists.py`的主要功能是对数据集生成符合PaddleRS格式要求的file list。使用方式如下：

```shell
python generate_file_lists.py --data_dir {数据集根目录路径} --save_dir {输出目录路径} [--subsets {数据集所包含子集名称}] [--subdirs {子目录名称}] [--glob_pattern {影像文件名匹配模板}] [--file_list_pattern {file list文件名模板}] [--store_abs_path] [--sep {file list中使用的分隔符}]
```

其中：

- `--data_dir`：数据集的根目录。
- `--save_dir`：保存生成的file list的目录。
- `--subsets`：数据集所包含子集名称。数据集中的影像应保存在`data_dir/subset/subdir/`或者`data_dir/subdir/` (当不指定`--subsets`时)，其中`subset`是通过`--subsets`指定的子集名称之一。示例：`--subsets train val test`。
- `--subdirs`：子目录名称。数据集中的影像应保存在`data_dir/subset/subdir/`或者`data_dir/subdir/` (当不指定`--subsets`时)，其中`subdir`是通过`--subdirs`指定的子目录名称之一。默认为`('images', 'masks')`。
- `--glob_pattern`：影像文件名匹配模板。默认为`*`，表示匹配所有文件。
- `--file_list_pattern`：file list文件名模板。默认为`'{subset}.txt'`。
- `--store_abs_path`：若指定此选项，则在file list中保存绝对路径，否则保存相对路径。
- `--sep`：file list中使用的分隔符，默认为` `（空格）。
