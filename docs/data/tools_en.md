# Remote Sensing Image Processing Toolkit

PaddleRS provides a rich set of remote sensing image processing tools in the `tools` directory, including:

- `coco2mask.py`: Convert COCO annotation files to .png files.
- `mask2shape.py`: Convert .png format raster labels from model inference output to .shp vector format.
- `geojson2mask.py`: Convert GeoJSON format labels to .tif raster format.
- `match.py`: Implement registration of two images.
- `split.py`: Split large-scale image data into tiles.
- `coco_tools/`: A collection of COCO tools for processing COCO format annotation files.
- `prepare_dataset/`: A collection of scripts for preprocessing datasets.
- `extract_ms_patches.py`: Extract multi-scale image blocks from entire remote sensing images.

## Usage

First, please make sure you have downloaded PaddleRS to your local machine. Navigate to the `tools` directory:

```shell
cd tools
```

### coco2mask

The main function of `coco2mask.py` is to convert images and corresponding COCO-formatted segmentation labels into images and labels in .png format, which are stored separately in the `img` and `gt` directories. The relevant data examples can be found in the [Chinese Typical City Building Instance Dataset](https://www.scidb.cn/detail?dataSetId=806674532768153600&dataSetType=journal). For the masks, the saved result is a single-channel pseudocolor image. The usage is as follows:

```shell
python coco2mask.py --raw_dir {input directory path} --save_dir {output directory path}
```

Among them:

- `raw_dir`: Directory where the raw data is stored. Images are stored in the `images` subdirectory, and labels are saved in the `xxx.json` format.
- `save_dir`: Directory where the output results are saved. Images are saved in the `img` subdirectory, and .png format labels are saved in the `gt` subdirectory.

### mask2shape

The main function of `mask2shape.py` is to convert the segmentation results in .png format into shapefile format (vector graphics). The usage is as follows:

```shell
python mask2shape.py --srcimg_path {path to the original image with geographic information} --mask_path {input segmentation label path} [--save_path {output vector graphics path}] [--ignore_index {index values to be ignored}]
```

Among them:

- `srcimg_path`: Path to the original image with geographic information, which is required to provide the shapefile with geoprojection coordinate system information.
- `mask_path`: Path to the .png format segmentation result obtained by the model inference.
- `save_path`: Path to save the shapefile. The default value is `output`.
- `ignore_index`: Index value to be ignored in the shapefile, such as the background class in segmentation tasks. The default value is `255`.

### geojson2mask

The main function of `geojson2mask.py` is to convert the GeoJson-formatted labels to a .tif raster format. The usage is as follows:

```shell
python geojson2mask.py --srcimg_path {path to the original image with geographic information} --geojson_path {input segmentation label path} --save_path {output path}
```

Among them:

- `srcimg_path`: Path to the original image file that contains the geospatial information.
- `geojson_path`: Path to the GeoJSON format label file.
- `save_path`: Path to save the converted raster file.

### match

The main function of `match.py` is to perform spatial registration on two temporal remote sensing images. The usage is as follows:

```shell
python match.py --im1_path [path to temporal image 1] --im2_path [path to temporal image 2] --save_path [output path for registered temporal image 2] [--im1_bands 1 2 3] [--im2_bands 1 2 3]
```

Among them:

- `im1_path`: File path of the first temporal image. This image must contain geospatial information and will be used as the reference image during the registration process.
- `im2_path`: File path of the second temporal image. The geospatial information of this image will not be used. This image will be registered to the first temporal image during the registration process.
- `im1_bands`: Bands of the first temporal image used for registration, specified as three channels (representing R, G, and B) or a single channel. Default is `[1, 2, 3]`.
- `im2_bands`: Bands of the second temporal image used for registration, specified as three channels (representing R, G, and B) or a single channel. Default is `[1, 2, 3]`.
- `save_path`: Output file path of the second temporal image after registration.

### split

The main function of `split.py` is to divide large remote sensing images into image blocks, which can be used as input for training. The usage is as follows:

```shell
python split.py --image_path {input image path} [--mask_path {Ground-truth label path}] [--block_size {image block size}] [--save_dir {output directory}]
```

Among them:

- `image_path`: Path of the image to be split.
- `mask_path`: Path of the label image to be split together. Default is `None`.
- `block_size`: Size of the split image blocks. Default is `512`.
- `save_dir`: Directory to save the split results. Default is `output`.

### coco_tools

There are six tools included in the `coco_tools` directory, each with the following functions

- `json_InfoShow.py`:    Print basic information of each dictionary in a json file.
- `json_ImgSta.py`:      Generate statistical table and graph of image information in a json file.
- `json_AnnoSta.py`:     Generate statistical table and graph of annotation information in a json file.
- `json_Img2Json.py`:    Generate a json file by counting images in a test set.
- `json_Split.py`:       Split the content of a json file into train set and val set.
- `json_Merge.py`:       Merge multiple json files into one.

For detailed usage instructions, please refer to [coco_tools Usage Instructions](coco_tools.md).

### prepare_dataset

The `prepare_dataset` directory contains a series of data preprocessing scripts that are mainly used to preprocess open source remote sensing datasets downloaded locally to meet the training, validation, and testing standards of PaddleRS.

Before executing the script, you can use the `--help` option to get help information. For example:

```shell
python prepare_dataset/prepare_levircd.py --help
```

The following are common command-line options in the script:

- `--in_dataset_dir`: Path to the downloaded original dataset on your local machine. Example: `--in_dataset_dir downloads/LEVIR-CD`.
- `--out_dataset_dir`:  Path to the processed dataset. Example: `--out_dataset_dir data/levircd`.
- `--crop_size`: For datasets that support image cropping, specify the size of the cropped image block. Example: `--crop_size 256`.
- `--crop_stride`: For datasets that support image cropping, specify the step size of the sliding window during cropping. Example: `--crop_stride 256`.
- `--seed`: Random seed. It can be used to fix the pseudo-random number sequence generated by the random number generator, so as to obtain a fixed dataset partitioning result. Example: `--seed 1919810`
- `--ratios`: For datasets that support random subset partitioning, specify the sample ratios of each subset that needs to be partitioned. Example: `--ratios 0.7 0.2 0.1`.

You can refer to [this document](https://github.com/PaddlePaddle/PaddleRS/blob/develop/docs/intro/data_prep.md) to see which preprocessing scripts for datasets are provided by PaddleRS.

### extract_ms_patches

The main function of `extract_ms_patches.py` is to extract image patches containing objects of interest at different scales from the entire remote sensing image using a quadtree. The extracted image patches can be used as training samples for models. The usage is as follows:

```shell
python extract_ms_patches.py --im_paths {one or more input image paths} --mask_path {Ground-truth label path} [--save_dir {output directory}] [--min_patch_size {minimum patch size}] [--bg_class {background class ID}] [--target_class {target class ID}] [--max_level {maximum level of scale}] [--include_bg] [--nonzero_ratio {threshold of the ratio of nonzero pixels}] [--visualize]
```

Among them:

- `im_paths`: Path of the source image(s). Multiple paths can be specified.
- `mask_path`: Path to the ground-truth label.
- `save_dir`: Path to the directory to save the split result. Default is `output`.
- `min_patch_size`: Minimum size of the extracted image block (in terms of the number of pixels in the length/width of the image block covered by the leaf nodes in the quadtree). Default is `256`.
- `bg_class`: Class ID of the background class. Default is `0`.
- `target_class`: Class ID of the target class. If it is `None`, it means that all classes except the background class are target classes. Default is `None`.
- `max_level`: Maximum level of scale to retrieve. If it is `None`, it means that there is no limit to the level. Default is `None`.
- `include_bg`: If specified, also save the image blocks that only contain the background class and do not contain the target class.
- `--nonzero_ratio`: Specify a threshold. For any source image, if the ratio of nonzero pixels in the image block is less than this threshold, the image block will be discarded. If it is `None`, no filtering will be performed. Default is `None`.
- `--visualize`: If specified, after the program is executed, the image `./vis_quadtree.png` will be generated, which visualizes the nodes in the quadtree. An example is shown in the following figure:

<div align="center">
<img src="https://user-images.githubusercontent.com/21275753/189264850-f94b3d7b-c631-47b1-9833-0800de2ccf54.png"  width = "400" />  
</div>
