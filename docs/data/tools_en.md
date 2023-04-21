[简体中文](tools_cn.md) | English

# Remote Sensing Image Processing Toolkit

PaddleRS provides a rich set of remote sensing image processing tools in the `tools` directory, including:

- `coco2mask.py`: Convert COCO annotation files to PNG files.
- `mask2shape.py`: Convert PNG format raster labels from model inference output to .shp vector format.
- `geojson2mask.py`: Convert GeoJSON format labels to .tif raster format.
- `match.py`: Implement registration of two images.
- `split.py`: Split large image into tiles.
- `coco_tools/`: A collection of COCO tools for processing COCO format annotation files.
- `prepare_dataset/`: A collection of scripts for preprocessing datasets.
- `extract_ms_patches.py`: Extract multi-scale image blocks from entire remote sensing images.
- `generate_file_lists.py`：Generate file lists.

## Usage

First, please make sure you have downloaded PaddleRS to your local machine. Navigate to the `tools` directory:

```shell
cd tools
```

### coco2mask

The main function of `coco2mask.py` is to convert images and corresponding COCO-formatted segmentation labels into images and labels in PNG format, which are stored separately in the `img` and `gt` directories. The relevant data examples can be found in the [Chinese Typical City Building Instance Dataset](https://www.scidb.cn/detail?dataSetId=806674532768153600&dataSetType=journal). For the masks, the saved result is a single-channel pseudo-color image. The usage is as follows:

```shell
python coco2mask.py --raw_dir {input directory path} --save_dir {output directory path}
```

Among them:

- `--raw_dir`: Directory where the raw data are stored. Images are stored in the `images` subdirectory, and labels are saved in the `xxx.json` format.
- `--save_dir`: Directory where the output results are saved. Images are saved in the `img` subdirectory, and PNG format labels are saved in the `gt` subdirectory.

### mask2shape

The main function of `mask2shape.py` is to convert the segmentation results in PNG format into shapefile format (vector graphics). The usage is as follows:

```shell
python mask2shape.py --src_img_path {path to the original image with geographic information} --mask_path {path to segmentation mask} [--save_path {path to save the output vector graphics}] [--ignore_index {index value to be ignored}]
```

Among them:

- `--src_img_path`: Path to the original image with geographic information, which is required to provide the shapefile with geoprojection coordinate system information.
- `--mask_path`: Path to the PNG format segmentation result obtained by the model inference.
- `--save_path`: Path to save the shapefile. The default value is `output`.
- `--ignore_index`: Index value to be ignored in the shapefile, such as the background class ID in segmentation tasks. The default value is `255`.

### geojson2mask

The main function of `geojson2mask.py` is to convert the GeoJSON-formatted labels to a .tif raster format. The usage is as follows:

```shell
python geojson2mask.py --src_img_path {path to the original image with geographic information} --geojson_path {path to segmentation mask} --save_path {output path}
```

Among them:

- `--src_img_path`: Path to the original image file that contains the geospatial information.
- `--geojson_path`: Path to the GeoJSON format label file.
- `--save_path`: Path to save the converted raster file.

### match

The main function of `match.py` is to perform spatial registration on two temporal remote sensing images. The usage is as follows:

```shell
python match.py --image1_path {path to temporal image 1} --image2_path {path to temporal image 2} --save_path {output path to registered image} [--image1_bands 1 2 3] [--image2_bands 1 2 3]
```

Among them:

- `--image1_path`: File path of the first temporal image. This image must contain geospatial information and will be used as the reference image during the registration process.
- `--image2_path`: File path of the second temporal image. The geospatial information of this image will not be used. This image will be registered to the first temporal image.
- `--image1_bands`: Bands of the first temporal image used for registration, specified as three channels (representing R, G, and B) or a single channel. Default is `[1, 2, 3]`.
- `--image2_bands`: Bands of the second temporal image used for registration, specified as three channels (representing R, G, and B) or a single channel. Default is `[1, 2, 3]`.
- `--save_path`: Output file path of the registered image.

### split

The main function of `split.py` is to divide large remote sensing images into image blocks. These image blocks can be stored and used as input for training. The usage is as follows:

```shell
python split.py --image_path {input image path} [--mask_path {ground-truth label image path}] [--block_size {image block size}] [--save_dir {output directory}]
```

Among them:

- `--image_path`: Path of the image to be split.
- `--mask_path`: Path of the ground-truth label image to be split together. Default is `None`.
- `--block_size`: Size of the image blocks. Default is `512`.
- `--save_dir`: Directory to save the cropped image blocks. Default is `output`.

### coco_tools

There are six tools included in the `coco_tools` directory, each with the following functions

- `json_info_show.py`:    Print basic information about each dictionary in the JSON file.
- `json_image_sta.py`:      Collect image information in JSON files and generate statistical tables and charts.
- `json_anno_sta.py`:     Collect annotation information in JSON files to generate statistical tables and charts.
- `json_image2json.py`:    Collect images of the test set and generate JSON file.
- `json_split.py`:       Split the JSON file into train set and val set.
- `json_merge.py`:       Merge multiple JSON files into one.

For detailed usage instructions, please refer to [coco_tools Usage Instructions](coco_tools_en.md).

### prepare_dataset

The `prepare_dataset` directory contains a series of data preprocessing scripts that are mainly used to preprocess open source remote sensing datasets to meet the training, validation, and testing standards of PaddleRS.

Before executing the script, you can use the `--help` option to get help information. For example:

```shell
python prepare_dataset/prepare_levircd.py --help
```

The following are common command-line options of the script:

- `--in_dataset_dir`: Path to the downloaded original dataset on your local machine. Example: `--in_dataset_dir downloads/LEVIR-CD`.
- `--out_dataset_dir`:  Path to the processed dataset. Example: `--out_dataset_dir data/levircd`.
- `--crop_size`: For datasets that support image cropping, specify the size of the cropped image block. Example: `--crop_size 256`.
- `--crop_stride`: For datasets that support image cropping, specify the step size of the sliding window during cropping. Example: `--crop_stride 256`.
- `--seed`: Random seed. It can be used to fix the pseudorandom number sequence generated by the random number generator, so as to obtain a deterministic dataset partitioning result. Example: `--seed 1919810`
- `--ratios`: For datasets that support random subset partitioning, specify the sample ratios of each subset that needs to be partitioned. Example: `--ratios 0.7 0.2 0.1`.

You can refer to [this document](../intro/data_prep_en.md) to see which preprocessing scripts for datasets are provided by PaddleRS.

### extract_ms_patches

The main function of `extract_ms_patches.py` is to extract image patches containing objects of interest at different scales from the entire remote sensing image using a quadtree. The extracted image patches can be used as training samples for deep learning models. The usage is as follows:

```shell
python extract_ms_patches.py --image_paths {one or more input image paths} --mask_path {ground-truth label image path} [--save_dir {output directory}] [--min_patch_size {minimum patch size}] [--bg_class {background category ID}] [--target_class {target category ID}] [--max_level {maximum scale level}] [--include_bg] [--nonzero_ratio {threshold of the ratio of nonzero pixels}] [--visualize]
```

Among them:

- `--image_paths`: Path of the source image(s). Multiple paths can be specified.
- `--mask_path`: Path to the ground-truth label.
- `--save_dir`: Path to the directory to save the split result. Default is `output`.
- `--min_patch_size`: Minimum size of the extracted image block (in terms of the number of pixels in the height/width of the image block). This is the minimum area covered by a leaf node in the quadtree. Default is `256`.
- `--bg_class`: Category ID of the background class. Default is `0`.
- `--target_class`: Category ID of the target class. If it is `None`, it means that all classes except the background class are target classes. Default is `None`.
- `--max_level`: Maximum scale level to retrieve. If it is `None`, it means that there is no limit to the scale level. Default is `None`.
- `--include_bg`: If specified, also save the image blocks that only contain the background class and do not contain the target class.
- `--nonzero_ratio`: Specify a threshold. For any source image, if the ratio of nonzero pixels in the image block is less than this threshold, the image block will be discarded. If it is `None`, no filtering will be performed. Default is `None`.
- `--visualize`: If specified, the image `./vis_quadtree.png` will be generated, which visualizes the nodes in the quadtree. An example is shown in the following figure:

<div align="center">
<img src="https://user-images.githubusercontent.com/21275753/189264850-f94b3d7b-c631-47b1-9833-0800de2ccf54.png"  width = "400" />  
</div>

### generate_file_lists

The main function of `generate_file_lists.py` is to generate file lists that contain the image and label paths of a dataset. The usage is as follows:

```shell
python generate_file_lists.py --data_dir {root directory of dataset} --save_dir {output directory} [--subsets {names of subsets}] [--subdirs {names of subdirectories}] [--glob_pattern {glob pattern used to match image files}] [--file_list_pattern {patterm to name the file lists}] [--store_abs_path] [--sep {delimeter to use in file lists}]
```

Among them:

- `--data_dir`: Root directory of the dataset.
- `--save_dir`: Directory to save the generated file lists.
- `--subsets`: Names of subsets. Images should be stored in `data_dir/subset/subdir/` or `data_dir/subdir/` (when `--subsets` is not specified), where `subset` is one of the values in `--subsets`. Example: `--subsets train val test`.
- `--subdirs`: Names of subdirectories. Images should be stored in `data_dir/subset/subdir/` or `data_dir/subdir/` (when `--subsets` is not specified), where `subdir` is one of the values in `--subdirs`. Defaults to `('images', 'masks')`.
- `--glob_pattern`: Glob pattern used to match image files. Defaults to `*`, which matches arbitrary file.
- `--file_list_pattern`: Pattern to name the file lists. Defaults to `{subset}.txt`.
- `--store_abs_path`: If specified, store the absolute path rather than the relative path in file lists.
- `--sep`: Delimiter to use when writing lines to file lists. Defaults to ` ` (a space).
