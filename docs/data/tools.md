# Remote sensing image processing toolset

PaddleRS offers a wealth of remote sensing image processing tools in the 'tools' directory, including:

- 'coco2mask.py' : Used to convert COCO annotation files to.png format.
- 'mask2shape.py' : Used to convert.png raster labels for model inference output to.shp vector format.
- 'geojson2mask.py' : Used to convert GeoJSON format tags to.tif raster format.
- 'match.py' : Used to register two images.
- 'split.py' : For slicing large format image data.
- 'coco_tools/' : COCO tools collection for statistical processing of COCO annotated files.
- 'prepare_dataset/' : a collection of dataset preprocessing scripts.
- 'extract_ms_patchs.py' : Extracts multi-scale patches from the whole remote sensing image.

## Instructions

Make sure you've downloaded PaddleRS locally first. Go to the 'tools' directory:

```shell
cd tools
```

### coco2mask

The main function of 'coco2mask.py' is to convert images and their COCO segmentation tags into images and.png tags. The results will be stored in two directories: 'img' and 'gt'. Related data sample can refer to [China city buildings typical instance data set] (https://www.scidb.cn/detail?dataSetId=806674532768153600&dataSetType=journal). For mask, the result is a single-channel false-color image. Here's how to use it:

```shell
python coco2mask.py --raw_dir {input directory path} --save_dir {output directory path}
` ` `

Among them:

- 'raw_dir' : The directory to store the raw data, where the images are stored in the 'images' subdirectory and the labels are saved in the' xxx.json 'format.
- 'save_dir' : The output directory where images are stored in the 'img' subdirectory and.png tags are stored in the 'gt' subdirectory.

### mask2shape

The main function of 'mask2shape.py' is to convert.png segmentation results to shapefile format (vector image). Here's how to use it:

```shell
python mask2shape.py --srcimg_path {raw image path with geography} --mask_path {input split label path} [--save_path {output vector path}] [--ignore_index {index values to ignore}]
```

Among them:

- 'srcimg_path' : The original image path with geolocation meta information to provide the generated shapefile with geolocation projection coordinates.
- 'mask_path' : Segmentation in.png format as a result of model inference.
- 'save_path' : The path to save the shapefile, defaults to 'output'.
- 'ignore_index' : Index values to ignore in the shapefile (e.g. background class in segmentation tasks), defaults to '255'.

### geojson2mask

The main function of 'geojson2mask.py' is to convert the GeoJSON tags to a.tif raster format. Here's how to use it:

```shell
python geojson2mask.py --srcimg_path {original image path with geographic information} --geojson_path {input split label path} --save_path {output path}
```
Among them:

- 'srcimg_path' : original image path, with geo-meta information required.
- 'geojson_path' : GeoJSON format tag path.
- 'save_path' : The path to save the converted raster file.

### match

The main function of 'match.py' is to perform spatial registration of remote sensing images in two temporal phases. Here's how to use it:

```shell
python match.py --im1_path [time 1 image path] --im2_path [time 2 image path] --save_path [output path of time 2 image after registration] [--im1_bands 1 2 3] [--im2_bands 1 2 3]
```

Among them:

- 'im1_path' : The phase 1 image path. This image must contain geographic information and is used as the reference image during registration.
- 'im2_path' : The phase 2 image path. The geographic information of the image will not be used. This image is registered to the time phase 1 image in the registration process.
- 'im1_bands' : bands used for registration of the phase 1 image, specified as three-channel (for R, G, B) or single-channel (default: [1, 2, 3]).
- 'im2_bands' : bands used for registration of the phase 2 image, specified as three channels (for R, G, B) or single channel, default to [1, 2, 3].
- 'save_path' : output path of phase 2 image after registration.

### split

The main function of 'split.py' is to divide large format remote sensing images into blocks that can be used as input for training. Here's how to use it:

```shell
python split.py --image_path {input image path} [--mask_path {Groundtruth label path}] [--block_size {image block size}] [--save_dir {output directory}]
```
Among them:

- 'image_path' : The path of the image to split.
- 'mask_path' : tag image paths to split together, defaults to 'None'.
- 'block_size' : Split image block size, default 512.
- 'save_dir' : The directory to store the split results. Default is' output '.

### coco_tools

There are currently 6 tools in the 'coco_tools' directory with the following functions:

- 'json_InfoShow.py' : Prints basic information about each dictionary in the json file;
- 'json_ImgSta.py' : Statistics the image information in the json file, generate statistical tables, statistical graphs;
- 'json_AnnoSta.py' : statistic the annotation information in the json file, generate statistical tables, statistical graphs;
- 'json_Img2Json.py' : generate json file with test images;
- 'json_Split.py' : splits the json file into a train set and a val set;
- 'json_Merge.py' : Merges multiple json files into one.

See [coco_tools Instructions](coco_tools.md) for details.

### prepare_dataset

The 'prepare_dataset' directory contains a series of data preprocessing scripts, which are mainly used to preprocess remote sensing open source datasets that have been downloaded locally to make them meet the standards of PaddleRS training, validation, and testing.

Before executing the script, you can get help with the '--help' option. For example:

```shell
python prepare_dataset/prepare_levircd.py --help
```

The following is a list of common command-line options in scripts:

- '--in_dataset_dir' : the path to the original dataset downloaded locally. Example: '--in_dataset_dir downloads/LEVIR-CD'.
- '--out_dataset_dir' : the path to the processed dataset. Example: '--out_dataset_dir data/levircd'.
- '--crop_size' : Specifies the chunk size to split into for a dataset that supports image clipping. Example: '--crop_size 256'.
- '--crop_stride' : Specifies the stride of the slider to move during splitting for datasets that support image clipping. Example: '--crop_stride 256'.
- '--seed' : random seed. It can be used to fix the pseudo-random number sequence generated by the random number generator, so as to obtain a fixed data set partition result. Example: '--seed 1919810'
- '--ratios' : For datasets that support random subset splitting, this specifies the proportion of samples that should be split into each subset. Example: '--ratios 0.7 0.2 0.1'.

You can view in [this document] (https://github.com/PaddlePaddle/PaddleRS/blob/develop/docs/intro/data_prep.md) PaddleRS provide which data sets the pretreatment of the script.

### extract_ms_patches

The main function of 'extract_ms_patchs.py' is to use a quadtree to extract patches of different scales containing the object of interest from the whole remote sensing image. The extracted patches can be used as training samples for the model. Here's how to use it:

```shell
python extract_ms_patchs.py --im_paths {one or more input image paths} --mask_path {truth label paths} [--save_dir {output directory}] [--min_patch_size {Minimum image block size}] [--bg_class {background class number}] [--target_class {target class number}] [--max_level {maximum scale level to retrieve}] [--include_bg] [--nonzero_ratio {Threshold of proportion of non-zero pixels in an image block}] [--visualize]
```

Among them:

- 'im_paths' : source image paths, multiple paths can be specified.
- 'mask_path' : truth label path.
- 'save_dir' : The directory to store the split results. Default is' output '.
- 'min_patch_size' : The minimum size of the extracted image patch (in pixels of patch length/width), i.e. the minimum area covered by the leaves of the quadtree in the graph. The default is' 256 '.
- 'bg_class' : The class number of the background class, defaults to '0'.
- 'target_class' : The class number of the target class. If it is' None ', it means that all classes other than the background class are the target class.
- 'max_level' : The maximum scale level to retrieve. If it is' None ', it means no level limit.
- 'include_bg' : If specified, we will also save blocks that contain only the background category and no target category.
- '--nonzero_ratio' : This specifies a threshold at which, for any source image, a block of non-zero_ratio will be discarded. A value of 'None' means no filtering. The default is' None '.
- '--visualize' : If this option is specified, the program will produce an image './vis_quadtree.png 'with a visualization of the nodes in the quadtree. An example is shown in the following figure:

<div align="center">
<img src="https://user-images.githubusercontent.com/21275753/189264850-f94b3d7b-c631-47b1-9833-0800de2ccf54.png"  width = "400" />  
</div>
