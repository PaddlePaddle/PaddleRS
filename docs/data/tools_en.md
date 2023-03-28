# Remote Sensing Image Processing Tool Set

PaddleRS provides a lot of remote sensing image processing tools in the `tools` directory, including:

- `coco2mask.py`: Convert COCO annotation files to .png files.
- `mask2shape.py`: Convert .png raster label output of model inference to .shp vector format.
- `geojson2mask.py`: Convert the GeoJSON format tag to a .tif raster format.
- `match.py`: Register two images.
- `split.py`: Slice a large image.
- `coco_tools/`: COCO tools collection for statistical processing of COCO format annotation files.
- `prepare_dataset/`: Collection of data set preprocessing scripts.
- `extract_ms_patches.py`: Extract multi-scale image blocks from a remote sensing image.

## Usage Instructions

Firstly make sure you have downloaded PaddleRS locally. Go to the `tools` directory:

```shell
cd tools
```

### coco2mask

The main function of `coco2mask.py` is to convert images and corresponding COCO-formatted segmentation labels into images and labels in .png format, which are stored separately in the `img` and `gt` directories. The relevant data examples can be found in the [Chinese Typical City Building Instance Dataset](https://www.scidb.cn/detail?dataSetId=806674532768153600&dataSetType=journal). For the masks, the saved result is a single-channel pseudocolor image. The usage is as follows:

```shell
python coco2mask.py --raw_dir {input directory path} --save_dir {output directory path}
```

Among them:

- `raw_dir`: The directory raw data stored, where the images are stored in the `images` subdirectory and the labels are stored in the `xxx.json` format.
- `save_dir`: The directory to save the output result, where images are stored in the `img` subdirectory and .png labels are stored in the `gt` subdirectory.

### mask2shape

The main function of `mask2shape.py` is to convert the segmentation results in .png format into shapefile format (vector image). The usage is as follows:

```shell
python mask2shape.py --srcimg_path {original image path with geographic information} --mask_path {input split label path} [--save_path {output vector map path}] [--ignore_index {index value to ignore}]
```

Among them:

- `srcimg_path`: The original image path needs to contain geographic meta information to provide the geographic projection coordinate system and other information for the generated shapefile.
- `mask_path`: .png format segmentation results obtained by model inference.
- `save_path`: Path to save shapefile. Default is `output`.
- `ignore_index`: Index value that need to be ignored in shapefile (such as background classes in split tasks) default to `255`.

### Geojson2mask

The main function of `geojson2mask.py` is to convert the geoJson-formatted tag to a .tif raster format. The usage is as follows:

```shell
python geojson2mask.py --srcimg_path {original image path with geographic information} --geojson_path {input split label path} --save_path {output path}
```

Among them:

- `srcimg_path`: The original image path requires geographic meta information.
- `geojson_path`: GeoJSON format label path.
- `save_path`: The path to save the converted raster file.

### match

The main function of `match.py` is to perform spatial registration for remote sensing images of two time phases. The usage is as follows:

```shell
python match.py --im1_path [time phase 1 image path] --im2_path [time phase 2 image path] --save_path [image output path of time phase 2 after registration] [--im1_bands 1 2 3] [--im2_bands 1 2 3]
```

Among them:

- `im1_path`: Time phase 1 image path. The image must contain geographic information and be used as the reference image during registration.
- `im2_path`: Time phase 2 image path. The geographic information of the image will not be used. In the registration process, the image is registered to phase 1 image.
- `im1_bands`: The band used for registration of the phase 1 image is specified as three-channel (representing R, G, B respectively) or single channel, default is [1, 2, 3].
- `im2_bands`: The band used for registration of time-phase 2 image is specified as three-channel (representing R, G, B respectively) or single channel. The default is [1, 2, 3].
- `save_path`: Image output path of time phase 2 after registration.

### split

The main function of `split.py` is to divide large-format remote sensing images into image blocks, which can be used as input in training. The usage is as follows:

```shell
python split.py --image_path {input image path} [--mask_path {GT label path}] [--block_size {image block size}] [--save_dir {output directory}]
```

Among them:

- `image_path`: The path of the image to be shred.
- `mask_path`: The label image path that is shred together defaults to `None`.
- `block_size`: Split image block size, default is 512.
- `save_dir`: Folder path to save the shred result, default is `output`.

### coco_tools

At present `coco_tools` directory contains 6 tools, each tool function is as follows:

- `json_InfoShow.py`:    Print basic information about each dictionary in the json file.
- `json_ImgSta.py`:      Collect image information in json files and generate statistical tables and charts.
- `json_AnnoSta.py`:     Collect annotation information in json files to generate statistical tables and charts.
- `json_Img2Json.py`:    Test set image statistics, generate json file;
- `json_Split.py`:       The contents of the json file are divided into train sets and val sets.
- `json_Merge.py`:       Merge multiple json files into one.

Please refer to [coco_tools Usage Instructions](coco_tools.md) for details.

### prepare_dataset

The directory `prepare_dataset` contains a series of data preprocessing scripts, which are mainly used to preprocess the remote sensing open source dataset that has been downloaded to the local area and make it conform to the standards of PaddleRS training, verification and testing.

Before executing the script, you can use the `--help` option to get help information. For example:

```shell
python prepare_dataset/prepare_levircd.py --help
```

The following lists common command-line options in scripts:

- `--in_dataset_dir`: The path where the original data set is downloaded to the local location. Example: `--in_dataset_dir downloads/LEVIR-CD`.
- `--out_dataset_dir`: Path to store the processed data set. Example: `--out_dataset_dir data/levircd`.
- `--crop_size`: For data sets that support image blocks, specify the image block size to be shred. Example: `--crop_size 256`.
- `--crop_stride`: For data sets that support image blocks, specify the step size of slide window movement during segmentation. Example: `--crop_stride 256`.
- `--seed`: Random seed. It can be used to fix pseudorandom number sequence generated by random number generator, so as to obtain fixed data set partitioning results. Example: `--seed 1919810`
- `--ratios`: For data sets that support random partitioning of subsets, specify the sample proportion of each subset that needs to be partitioned. Example:`--ratios 0.7 0.2 0.1`.

You can see the preprocessing scripts for which data sets PaddleRS provides in [this document](https://github.com/PaddlePaddle/PaddleRS/blob/develop/docs/intro/data_prep.md).

### extract_ms_patches

The main function of `extract_ms_patches.py` is to use quadtree to extract image blocks containing objects of interest at different scales from the whole remote sensing image, and the extracted image blocks can be used as model training samples. The usage is as follows:

```shell
python extract_ms_patches.py --im_paths {One or more input image paths} --mask_path {GT label path} [--save_dir {output directory}] [--min_patch_size {Minimum image block size}] [--bg_class {Background class Category number}] [--target_class {Target class category number}] [--max_level {Maximum scale level of retrieval}] [--include_bg] [--nonzero_ratio {Non-zero pixel ratio threshold of image block}] [--visualize]
```

Among them:

- `im_paths`: Source image path, more than one path can be specified.
- `mask_path`: GT label path.
- `save_dir`: Folder path to save the shred result, default is `output`.
- `min_patch_size`: The minimum size of the extracted image block (in pixels of image block length/width), that is, the minimum range of leaf nodes in the quadtree covered in the image, defaults to `256`.
- `bg_class`: Category number of the background category. The default is `0`.
- `target_class`: Category number of the target category. If it is `None`, it means that all categories other than the background category are target categories. The default is `None`.
- `max_level`: Maximum scale level of retrieval, if `None`, means unrestricted level, default is `None`.
- `include_bg`: If this option is specified, the image blocks that contain only the background category and not the target category are also saved.
- `--nonzero_ratio`: Specifies a threshold. For any source image, if the proportion of non-zero pixels in the image block is less than this threshold, the image block will be abandoned. If the value is `None`, no filtering is performed. The default is `None`.
- `--visualize`: If this option is specified, an image `./vis_quadtree.png` will be generated upon completion of the program, which will store the visual result of the node condition in the quadtree. An example is shown below:

<div align="center">
<img src="https://user-images.githubusercontent.com/21275753/189264850-f94b3d7b-c631-47b1-9833-0800de2ccf54.png"  width = "400" />  
</div>
