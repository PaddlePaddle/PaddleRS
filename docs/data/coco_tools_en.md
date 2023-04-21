[简体中文](coco_tools_cn.md) | English

# coco_tools Instructions

## 1 Tool Description

coco_tools is a set of tools provided by PaddleRS for handling COCO annotation files. It is located in the `tools/coco_tools/` directory. Because [pycocotools library] (https://pypi.org/project/pycocotools/) cannot be installed in some environment, PaddleRS provides coco_tools as an alternative.

## 2 Document Description

At present, coco_tools has 6 files, in the following we show each file and its function:

- `json_info_show.py`:    Print basic information about each dictionary in the JSON file.
- `json_image_sta.py`:      Collect image information in JSON files and generate statistical tables and charts.
- `json_anno_sta.py`:     Collect annotation information in JSON files to generate statistical tables and charts.
- `json_image2json.py`:    Collect images of the test set and generate JSON file.
- `json_split.py`:       Split the JSON file into train set and val set.
- `json_merge.py`:       Merge multiple JSON files into one.

## 3 Usage Example

### 3.1 Sample Dataset

This document uses the COCO 2017 dataset as sample data to demonstrate. You can download the dataset at the following link:

- [Official download link](https://cocodataset.org/#download)
- [aistudio backup link](https://aistudio.baidu.com/aistudio/datasetdetail/7122)

After the download is complete, you can copy or link the `coco_tools` directory from the PaddleRS project to the dataset directory for future use. The complete data set directory structure is as follows:

```
./COCO2017/      # dataset root directory
|--train2017     # training dataset original image directory
|  |--...
|  |--...
|--val2017       # validation dataset original image directory
|  |--...
|  |--...
|--test2017      # test dataset original image directory
|  |--...
|  |--...
|
|--annotations   # annotation files directory
|  |--...
|  |--...
|
|--coco_tools    # coco_tools code directory
|  |--...
|  |--...
```

### 3.2 Printing json Information

Using `json_info_show.py` allows you to print the keys of each key-value pair in a JSON file and output the first element in the value to help you quickly understand the annotation information. For COCO format annotations, you should pay special attention to the contents of the `image` and `annotation` fields.

#### 3.2.1 Command Demonstration

Run the following command to print the information in `instances_val2017.json` :

```
python ./coco_tools/json_info_show.py \
       --json_path=./annotations/instances_val2017.json \
       --show_num 5
```

#### 3.2.2 Parameter Description


| Parameter Name | Description                                                 | Default Value |
| -------------- | ----------------------------------------------------------- | ------------- |
| `--json_path`  | Path of the JSON file whose statistics are to be collected. |               |
| `--show_num`   | (Optional) Number of elements to show in the output.        | `5`           |

#### 3.2.3 Result Presentation

After the preceding command is executed, the following information will be displayed:

```
------------------------------------------------Args------------------------------------------------
json_path = ./annotations/instances_val2017.json
show_num = 5

------------------------------------------------Info------------------------------------------------
json read...
json keys: dict_keys(['info', 'licenses', 'images', 'annotations', 'categories'])

***********************info***********************
 Content Type: dict
 Total Length: 6
 First 5 record:

description : COCO 2017 Dataset
url : http://cocodataset.org
version : 1.0
year : 2017
contributor : COCO Consortium
...
...

*********************licenses*********************
 Content Type: list
 Total Length: 8
 First 5 record:

{'url': 'http://creativecommons.org/licenses/by-nc-sa/2.0/', 'id': 1, 'name': 'Attribution-NonCommercial-ShareAlike License'}
{'url': 'http://creativecommons.org/licenses/by-nc/2.0/', 'id': 2, 'name': 'Attribution-NonCommercial License'}
{'url': 'http://creativecommons.org/licenses/by-nc-nd/2.0/', 'id': 3, 'name': 'Attribution-NonCommercial-NoDerivs License'}
{'url': 'http://creativecommons.org/licenses/by/2.0/', 'id': 4, 'name': 'Attribution License'}
{'url': 'http://creativecommons.org/licenses/by-sa/2.0/', 'id': 5, 'name': 'Attribution-ShareAlike License'}
...
...

**********************images**********************
 Content Type: list
 Total Length: 5000
 First 5 record:

{'license': 4, 'file_name': '000000397133.jpg', 'coco_url': 'http://images.cocodataset.org/val2017/000000397133.jpg', 'height': 427, 'width': 640, 'date_captured': '2013-11-14 17:02:52', 'flickr_url': 'http://farm7.staticflickr.com/6116/6255196340_da26cf2c9e_z.jpg', 'id': 397133}
{'license': 1, 'file_name': '000000037777.jpg', 'coco_url': 'http://images.cocodataset.org/val2017/000000037777.jpg', 'height': 230, 'width': 352, 'date_captured': '2013-11-14 20:55:31', 'flickr_url': 'http://farm9.staticflickr.com/8429/7839199426_f6d48aa585_z.jpg', 'id': 37777}
{'license': 4, 'file_name': '000000252219.jpg', 'coco_url': 'http://images.cocodataset.org/val2017/000000252219.jpg', 'height': 428, 'width': 640, 'date_captured': '2013-11-14 22:32:02', 'flickr_url': 'http://farm4.staticflickr.com/3446/3232237447_13d84bd0a1_z.jpg', 'id': 252219}
{'license': 1, 'file_name': '000000087038.jpg', 'coco_url': 'http://images.cocodataset.org/val2017/000000087038.jpg', 'height': 480, 'width': 640, 'date_captured': '2013-11-14 23:11:37', 'flickr_url': 'http://farm8.staticflickr.com/7355/8825114508_b0fa4d7168_z.jpg', 'id': 87038}
{'license': 6, 'file_name': '000000174482.jpg', 'coco_url': 'http://images.cocodataset.org/val2017/000000174482.jpg', 'height': 388, 'width': 640, 'date_captured': '2013-11-14 23:16:55', 'flickr_url': 'http://farm8.staticflickr.com/7020/6478877255_242f741dd1_z.jpg', 'id': 174482}
...
...

*******************annotations********************
 Content Type: list
 Total Length: 36781
 First 5 record:

{'segmentation': [[510.66, 423.01, 511.72, 420.03, 510.45, 416.0, 510.34, 413.02, 510.77, 410.26, 510.77, 407.5, 510.34, 405.16, 511.51, 402.83, 511.41, 400.49, 510.24, 398.16, 509.39, 397.31, 504.61, 399.22, 502.17, 399.64, 500.89, 401.66, 500.47, 402.08, 499.09, 401.87, 495.79, 401.98, 490.59, 401.77, 488.79, 401.77, 485.39, 398.58, 483.9, 397.31, 481.56, 396.35, 478.48, 395.93, 476.68, 396.03, 475.4, 396.77, 473.92, 398.79, 473.28, 399.96, 473.49, 401.87, 474.56, 403.47, 473.07, 405.59, 473.39, 407.71, 476.68, 409.41, 479.23, 409.73, 481.56, 410.69, 480.4, 411.85, 481.35, 414.93, 479.86, 418.65, 477.32, 420.03, 476.04, 422.58, 479.02, 422.58, 480.29, 423.01, 483.79, 419.93, 486.66, 416.21, 490.06, 415.57, 492.18, 416.85, 491.65, 420.24, 492.82, 422.9, 493.56, 424.39, 496.43, 424.6, 498.02, 423.01, 498.13, 421.31, 497.07, 420.03, 497.07, 415.15, 496.33, 414.51, 501.1, 411.96, 502.06, 411.32, 503.02, 415.04, 503.33, 418.12, 501.1, 420.24, 498.98, 421.63, 500.47, 424.39, 505.03, 423.32, 506.2, 421.31, 507.69, 419.5, 506.31, 423.32, 510.03, 423.01, 510.45, 423.01]], 'area': 702.1057499999998, 'iscrowd': 0, 'image_id': 289343, 'bbox': [473.07, 395.93, 38.65, 28.67], 'category_id': 18, 'id': 1768}
{'segmentation': [[289.74, 443.39, 302.29, 445.32, 308.09, 427.94, 310.02, 416.35, 304.23, 405.73, 300.14, 385.01, 298.23, 359.52, 295.04, 365.89, 282.3, 362.71, 275.29, 358.25, 277.2, 346.14, 280.39, 339.13, 284.85, 339.13, 291.22, 338.49, 293.77, 335.95, 295.04, 326.39, 297.59, 317.47, 289.94, 309.82, 287.4, 288.79, 286.12, 275.41, 284.21, 271.59, 279.11, 276.69, 275.93, 275.41, 272.1, 271.59, 274.01, 267.77, 275.93, 265.22, 277.84, 264.58, 282.3, 251.2, 293.77, 238.46, 307.79, 221.25, 314.79, 211.69, 325.63, 205.96, 338.37, 205.32, 347.29, 205.32, 353.03, 205.32, 361.31, 200.23, 367.95, 202.02, 372.27, 205.8, 382.52, 215.51, 388.46, 225.22, 399.25, 235.47, 399.25, 252.74, 390.08, 247.34, 386.84, 247.34, 388.46, 256.52, 397.09, 268.93, 413.28, 298.6, 421.91, 356.87, 424.07, 391.4, 422.99, 409.74, 420.29, 428.63, 415.43, 433.48, 407.88, 414.6, 405.72, 391.94, 401.41, 404.89, 394.39, 420.54, 391.69, 435.64, 391.15, 447.51, 387.38, 461.0, 384.68, 480.0, 354.47, 477.73, 363.1, 433.48, 370.65, 405.43, 369.03, 394.64, 361.48, 398.95, 355.54, 403.81, 351.77, 403.81, 343.68, 403.27, 339.36, 402.19, 335.58, 404.89, 333.42, 411.9, 332.34, 416.76, 333.42, 425.93, 334.5, 430.79, 336.12, 435.64, 321.01, 464.78, 316.16, 468.01, 307.53, 472.33, 297.28, 472.33, 290.26, 471.25, 285.94, 472.33, 283.79, 464.78, 280.01, 462.62, 284.33, 454.53, 285.94, 453.45, 282.71, 448.59, 288.64, 444.27, 291.88, 443.74]], 'area': 27718.476299999995, 'iscrowd': 0, 'image_id': 61471, 'bbox': [272.1, 200.23, 151.97, 279.77], 'category_id': 18, 'id': 1773}
{'segmentation': [[147.76, 396.11, 158.48, 355.91, 153.12, 347.87, 137.04, 346.26, 125.25, 339.29, 124.71, 301.77, 139.18, 262.64, 159.55, 232.63, 185.82, 209.04, 226.01, 196.72, 244.77, 196.18, 251.74, 202.08, 275.33, 224.59, 283.9, 232.63, 295.16, 240.67, 315.53, 247.1, 327.85, 249.78, 338.57, 253.0, 354.12, 263.72, 379.31, 276.04, 395.39, 286.23, 424.33, 304.99, 454.95, 336.93, 479.62, 387.02, 491.58, 436.36, 494.57, 453.55, 497.56, 463.27, 493.08, 511.86, 487.02, 532.62, 470.4, 552.99, 401.26, 552.99, 399.65, 547.63, 407.15, 535.3, 389.46, 536.91, 374.46, 540.13, 356.23, 540.13, 354.09, 536.91, 341.23, 533.16, 340.15, 526.19, 342.83, 518.69, 355.7, 512.26, 360.52, 510.65, 374.46, 510.11, 375.53, 494.03, 369.1, 497.25, 361.06, 491.89, 361.59, 488.67, 354.63, 489.21, 346.05, 496.71, 343.37, 492.42, 335.33, 495.64, 333.19, 489.21, 327.83, 488.67, 323.0, 499.39, 312.82, 520.83, 304.24, 531.02, 291.91, 535.84, 273.69, 536.91, 269.4, 533.7, 261.36, 533.7, 256.0, 531.02, 254.93, 524.58, 268.33, 509.58, 277.98, 505.82, 287.09, 505.29, 301.56, 481.7, 302.1, 462.41, 294.06, 481.17, 289.77, 488.14, 277.98, 489.74, 261.36, 489.21, 254.93, 488.67, 254.93, 484.38, 244.75, 482.24, 247.96, 473.66, 260.83, 467.23, 276.37, 464.02, 283.34, 446.33, 285.48, 431.32, 287.63, 412.02, 277.98, 407.74, 260.29, 403.99, 257.61, 401.31, 255.47, 391.12, 233.8, 389.37, 220.18, 393.91, 210.65, 393.91, 199.76, 406.61, 187.51, 417.96, 178.43, 420.68, 167.99, 420.68, 163.45, 418.41, 158.01, 419.32, 148.47, 418.41, 145.3, 413.88, 146.66, 402.53]], 'area': 78969.31690000003, 'iscrowd': 0, 'image_id': 472375, 'bbox': [124.71, 196.18, 372.85, 356.81], 'category_id': 18, 'id': 2551}
{'segmentation': [[260.4, 231.26, 215.06, 274.01, 194.33, 307.69, 195.63, 329.72, 168.42, 355.63, 120.49, 382.83, 112.71, 415.22, 159.35, 457.98, 172.31, 483.89, 229.31, 504.62, 275.95, 500.73, 288.91, 495.55, 344.62, 605.67, 395.14, 634.17, 480.0, 632.87, 480.0, 284.37, 404.21, 223.48, 336.84, 202.75, 269.47, 154.82, 218.95, 179.43, 203.4, 194.98, 190.45, 211.82, 233.2, 205.34]], 'area': 108316.66515000002, 'iscrowd': 0, 'image_id': 520301, 'bbox': [112.71, 154.82, 367.29, 479.35], 'category_id': 18, 'id': 3186}
{'segmentation': [[200.61, 253.97, 273.19, 318.49, 302.43, 336.64, 357.87, 340.67, 402.23, 316.48, 470.78, 331.6, 521.19, 321.52, 583.69, 323.53, 598.81, 287.24, 600.83, 236.84, 584.7, 190.46, 580.66, 169.29, 531.27, 121.91, 472.8, 93.69, 420.38, 89.65, 340.74, 108.81, 295.37, 119.9, 263.11, 141.07, 233.88, 183.41, 213.72, 229.78, 200.61, 248.93]], 'area': 75864.53530000002, 'iscrowd': 0, 'image_id': 579321, 'bbox': [200.61, 89.65, 400.22, 251.02], 'category_id': 18, 'id': 3419}
...
...

********************categories********************
 Content Type: list
 Total Length: 80
 First 5 record:

{'supercategory': 'person', 'id': 1, 'name': 'person'}
{'supercategory': 'vehicle', 'id': 2, 'name': 'bicycle'}
{'supercategory': 'vehicle', 'id': 3, 'name': 'car'}
{'supercategory': 'vehicle', 'id': 4, 'name': 'motorcycle'}
{'supercategory': 'vehicle', 'id': 5, 'name': 'airplane'}
...
...

```

#### 3.2.4 Result Description

`instances_val2017.json` has 5 keys:

```
'info', 'licenses', 'images', 'annotations', 'categories'
```

Among them,

- `'info'`: A dictionary. There are 6 key-value pairs, and the output shows the first five pairs.
- `'licenses'`: A list with 8 elements, and the output shows the first five.
- `'images'`: A list with 5000 elements, and the output shows the first five.
- `'annotations'`: A list with 36,781 elements, and the output shows the first five.
- `'categories'`: A list of 80 elements, and the output shows the first five.

### 3.3 Statistical Image Information

Using `json_image_sta.py`, you can quickly extract image information from `instances_val2017.json`, generate csv tables, and generate statistical graphs.

#### 3.3.1 Command Demonstration

Run the following command to print the information of `instances_val2017.json`:

```
python ./coco_tools/json_image_sta.py \
    --json_path=./annotations/instances_val2017.json \
    --csv_path=./img_sta/images.csv \
    --img_shape_path=./img_sta/img_shape.png \
    --img_shape_rate_path=./img_sta/img_shape_rate.png
```

#### 3.3.2 Parameter Description


| Parameter Name          | Description                                                                                                                           | Default Value |
|-------------------------| ------------------------------------------------------------------------------------------------------------------------------------- | ------------- |
| `--json_path`           | Path of the JSON file whose statistics are to be collected.                                                                           |               |
| `--csv_path`            | (Optional) Path for the statistics table.                                                                                             | `None`        |
| `--img_shape_path`      | (Optional) Output image saving path. The image visualizes the two-dimensional distribution of all image shapes.                         | `5`           |
| `--img_shape_rate_path` | (Optional) Output image saving path. The image visualizes the one-dimensional distribution of shape ratio (width/height) of all images. | `5`           |
| `--img_keyname`         | (Optional) Image key in the JSON file.                                                                                                | `'images'`    |

#### 3.3.3 Result Presentation

After the preceding command is executed, the following information will be displayed:

```
------------------------------------------------Args------------------------------------------------
json_path = ./annotations/instances_val2017.json
csv_path = ./img_sta/img.csv
img_shape_path = ./img_sta/img_shape.png
img_shape_rate_path = ./img_sta/img_shape_rate.png
img_keyname = images

json read...

make dir: ./img_sta
png save to ./img_sta/img_shape.png
png save to ./img_sta/img_shape_rate.png
csv save to ./img_sta/img.csv
```

Part of the table:


|   | license | file_name        | coco_url                                               | height | width | date_captured       | flickr_url                                                     | id     | shape_rate |
| - | ------- | ---------------- | ------------------------------------------------------ | ------ | ----- | ------------------- | -------------------------------------------------------------- | ------ | ---------- |
| 0 | 4       | 000000397133.jpg | http://images.cocodataset.org/val2017/000000397133.jpg | 427    | 640   | 2013-11-14 17:02:52 | http://farm7.staticflickr.com/6116/6255196340_da26cf2c9e_z.jpg | 397133 | 1.5        |
| 1 | 1       | 000000037777.jpg | http://images.cocodataset.org/val2017/000000037777.jpg | 230    | 352   | 2013-11-14 20:55:31 | http://farm9.staticflickr.com/8429/7839199426_f6d48aa585_z.jpg | 37777  | 1.5        |
| 2 | 4       | 000000252219.jpg | http://images.cocodataset.org/val2017/000000252219.jpg | 428    | 640   | 2013-11-14 22:32:02 | http://farm4.staticflickr.com/3446/3232237447_13d84bd0a1_z.jpg | 252219 | 1.5        |
| 3 | 1       | 000000087038.jpg | http://images.cocodataset.org/val2017/000000087038.jpg | 480    | 640   | 2013-11-14 23:11:37 | http://farm8.staticflickr.com/7355/8825114508_b0fa4d7168_z.jpg | 87038  | 1.3        |

Contents of saved pictures:

Two-dimensional distribution of all image shapes:
![image.png](./assets/1650011491220-image.png)

One-dimensional distribution of shape ratio (width/height) of all images:
![image.png](./assets/1650011634205-image.png)

### 3.4 Collect Statistics about the Object Detection Label Box

Using `json_anno_sta.py`, you can quickly extract annotation information from `instances_val2017.json`, generate csv tables, and generate statistical graphs.

#### 3.4.1 Command Demonstration

Run the following command to print the information of `instances_val2017.json`:

```
python ./coco_tools/json_anno_sta.py \
    --json_path=./annotations/instances_val2017.json \
    --csv_path=./anno_sta/obj.csv \
    --obj_shape_path=./anno_sta/obj_shape.png \
    --obj_shape_rate_path=./anno_sta/obj_shape_rate.png \
    --obj_pos_path=./anno_sta/obj_pos.png \
    --obj_pos_end_path=./anno_sta/obj_pos_end.png \
    --obj_cat_path=./anno_sta/obj_cat.png \
    --obj_num_path=./anno_sta/obj_num.png \
    --get_relative=True
```

#### 3.4.2 Parameter Description


| Parameter Name          | Description                                                                                                                                                                                                                                                                             | Default Value   |
|-------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------|
| `--json_path`           | Path of the JSON file whose statistics you want to collect.                                                                                                                                                                                                                             |                 |
| `--csv_path`            | (Optional) Path to save the statistics table.                                                                                                                                                                                                                                           | `None`          |
| `--obj_shape_path`      | (Optional) Output image saving path. The image visualizes the two-dimensional distribution of the shape of all object detection boxes.                                                                                                                                                    | `None`          |
| `--obj_shape_rate_path` | (Optional) Output image saving path. The image visualizes the one-dimensional distribution of shape ratio (width/height) of all target bounding boxes.                                                                                                                                    | `None`          |
| `--obj_pos_path`        | (Optional) Output image saving path. The image visualizes the two-dimensional distribution of the coordinates at the upper left corner of all bounding boxes.                                                                                                                             | `None`          |
| `--obj_pos_end_path`    | (Optional) Output image saving path. The image visualizes the two-dimensional distribution of the coordinates at the lower right corner of all bounding boxes.                                                                                                                            | `None`          |
| `--obj_cat_path`        | (Optional) Output image saving path. The image visualizes the quantity distribution of objects in each category.                                                                                                                                                                          | `None`          |
| `--obj_obj_num_path`    | (Optional) Output image saving path. The image visualizes the quantity distribution of annotated objects in a single image.                                                                                                                                                               | `None`          |
| `--get_relative`        | (Optional) Whether to generate the shape of the image target detection frame and the relative ratio of the coordinates of the upper left corner and lower right corner of the object detection frame (horizontal axis coordinates/image length, vertical axis coordinates/image width). | `False`         |
| `--img_keyname`         | (Optional) Image key in the JSON file.                                                                                                                                                                                                                                                  | `'images'`      |
| `--anno_keyname`        | (Optional) Annotation key in the JSON file.                                                                                                                                                                                                                                             | `'annotations'` |

#### 3.4.3 Result Presentation

After the preceding command is executed, the following information will be displayed:

```
------------------------------------------------Args------------------------------------------------
json_path = ./annotations/instances_val2017.json
csv_path = ./anno_sta/obj.csv
obj_shape_path = ./anno_sta/obj_shape.png
obj_shape_rate_path = ./anno_sta/obj_shape_rate.png
obj_pos_path = ./anno_sta/obj_pos.png
obj_pos_end_path = ./anno_sta/obj_pos_end.png
obj_cat_path = ./anno_sta/obj_cat.png
obj_obj_num_path = ./anno_sta/obj_obj_num.png
get_relative = True
img_keyname = images
anno_keyname = annotations

json read...

make dir: ./anno_sta
png save to ./anno_sta/obj_shape.png
png save to ./anno_sta/obj_shape_relative.png
png save to ./anno_sta/obj_shape_rate.png
png save to ./anno_sta/obj_pos.png
png save to ./anno_sta/obj_pos_relative.png
png save to ./anno_sta/obj_pos_end.png
png save to ./anno_sta/obj_pos_end_relative.png
png save to ./anno_sta/obj_cat.png
png save to ./anno_sta/obj_num.png
csv save to ./anno_sta/obj.csv
```

Part of the table:

![image.png](./assets/1650025881244-image.png)

Two-dimensional distribution of shape of all object bounding boxes:

![image.png](./assets/1650025909461-image.png)

Two-dimensional distribution of relative proportions of all object detection box shapes in the image:

![image.png](./assets/1650026052596-image.png)

One-dimensional distribution of shape ratio (width/height) of all object bounding boxes:

![image.png](./assets/1650026072233-image.png)

Two-dimensional distribution of coordinates at the upper left corner of all object bounding boxes:

![image.png](./assets/1650026247150-image.png)

Two-dimensional distribution of relative proportional values of coordinates at the upper left corner of all object bounding boxes:

![image.png](./assets/1650026289987-image.png)

Two-dimensional distribution of coordinates at the lower right corner of all object bounding boxes:

![image.png](./assets/1650026457254-image.png)

Two-dimensional distribution of relative proportional values of coordinates at the lower right corner of all object bounding boxes:

![image.png](./assets/1650026487732-image.png)

Distribution of the number of objects in each category:

![image.png](./assets/1650026546304-image.png)

A single image contains the quantity distribution of annotated objects:

![image.png](./assets/1650026559309-image.png)

### 3.5 Stat Image Information to Generates json

Using `json_image2json.py`, you can quickly extract image information according to the file information in `test2017` and the training set JSON file, and generate the test set JSON file.

#### 3.5.1 Command Demonstration

Run the following command to collect and generate `test2017` information:

```
python ./coco_tools/json_image2json.py \
    --image_dir=./test2017 \
    --json_train_path=./annotations/instances_val2017.json \
    --json_test_path=./test.json
```

#### 3.5.2 Parameter Description


| Parameter Name      | Description                                            | Default Value  |
|---------------------|--------------------------------------------------------| -------------- |
| `--image_dir`       | Directory that contains the test set images.           |                |
| `--json_train_path` | JSON file path of the training set. Used as reference. |                |
| `--json_test_path`  | Path to the generated test set JSON file.              |                |
| `--img_keyname`     | (Optional) Image key in the JSON file.                 | `'images'`     |
| `--cat_keyname`     | (Optional) Category key in the JSON file.              | `'categories'` |

#### 3.5.3 Result Presentation

After the preceding command is executed, the following information will be displayed:

```
------------------------------------------------Args------------------------------------------------
image_dir = ./test2017
json_train_path = ./annotations/instances_val2017.json
json_test_path = ./test.json

----------------------------------------------Get Test----------------------------------------------

json read...

test image read...
100%|█████████████████████████████████████| 40670/40670 [06:48<00:00, 99.62it/s]

 total test image: 40670
```

The generated JSON file information:

```
------------------------------------------------Args------------------------------------------------
json_path = ./test.json
show_num = 5

------------------------------------------------Info------------------------------------------------
json read...
json keys: dict_keys(['images', 'categories'])

**********************images**********************
 Content Type: list
 Total Length: 40670
 First 5 record:

{'id': 0, 'width': 640, 'height': 427, 'file_name': '000000379269.jpg'}
{'id': 1, 'width': 640, 'height': 360, 'file_name': '000000086462.jpg'}
{'id': 2, 'width': 640, 'height': 427, 'file_name': '000000176710.jpg'}
{'id': 3, 'width': 640, 'height': 426, 'file_name': '000000071106.jpg'}
{'id': 4, 'width': 596, 'height': 640, 'file_name': '000000251918.jpg'}
...
...

********************categories********************
 Content Type: list
 Total Length: 80
 First 5 record:

{'supercategory': 'person', 'id': 1, 'name': 'person'}
{'supercategory': 'vehicle', 'id': 2, 'name': 'bicycle'}
{'supercategory': 'vehicle', 'id': 3, 'name': 'car'}
{'supercategory': 'vehicle', 'id': 4, 'name': 'motorcycle'}
{'supercategory': 'vehicle', 'id': 5, 'name': 'airplane'}
...
...
```

### 3.6 json File Splitting

Using `json_split.py`, you can split the `instances_val2017.json` file into 2 subsets.

#### 3.6.1 Command Demonstration

Run the following command to split the `instances_val2017.json` file:

```
python ./coco_tools/json_split.py \
    --json_all_path=./annotations/instances_val2017.json \
    --json_train_path=./instances_val2017_train.json \
    --json_val_path=./instances_val2017_val.json
```

#### 3.6.2 Parameter Description


| Parameter Name        | Description                                                                                          | Default Value  |
|-----------------------| ---------------------------------------------------------------------------------------------------- | -------------- |
| `--json_all_path`     | Path to the original JSON file.                                                                      |                |
| `--json_train_path`   | Generated JSON file for the train set.                                                               |                |
| `--json_val_path`     | Generated JSON file for the val set.                                                                 |                |
| `--val_split_rate`    | (Optional) Proportion of files in the val set.                                                       | `0.1`          |
| `--val_split_num`     | (Optional) Number of val set files. If this parameter is set,`--val_split_rate` will be invalidated. | `None`         |
| `--keep_val_in_train` | (Optional) Whether to keep the val set samples in the train set.                                     | `False`        |
| `--img_keyname`       | (Optional) Image key in the JSON file.                                                               | `'images'`     |
| `--cat_keyname`       | (Optional) Category key in the JSON file.                                                            | `'categories'` |

#### 3.6.3 Result Presentation

After the preceding command is executed, the following information will be displayed:

```
------------------------------------------------Args------------------------------------------------
json_all_path = ./annotations/instances_val2017.json
json_train_path = ./instances_val2017_train.json
json_val_path = ./instances_val2017_val.json
val_split_rate = 0.1
val_split_num = None
keep_val_in_train = False
img_keyname = images
anno_keyname = annotations

-----------------------------------------------Split------------------------------------------------

json read...

image total 5000, train 4500, val 500
anno total 36781, train 33119, val 3662
```

### 3.7 json File Merging

Using `json_merge.py` to merge two JSON files.

#### 3.7.1 Command Demonstration

Run the following command to merge `instances_train2017.json` and `instances_val2017.json`:

```
python ./coco_tools/json_merge.py \
    --json1_path=./annotations/instances_train2017.json \
    --json2_path=./annotations/instances_val2017.json \
    --save_path=./instances_trainval2017.json
```

#### 3.7.2 Parameter Description


| Parameter Name | Description                                              | Default Value               |
| -------------- | -------------------------------------------------------- | --------------------------- |
| `--json1_path` | Path of the first JSON file to merge.                    |                             |
| `--json2_path` | Path of the second JSON file to merge.                   |                             |
| `--save_path`  | Path to save the merged JSON file.                       |                             |
| `--merge_keys` | (Optional) Keys to be merged.                            | `['images', 'annotations']` |

#### 3.7.3 Result Presentation

After the preceding command is executed, the following information will be displayed:

```
------------------------------------------------Args------------------------------------------------
json1_path = ./annotations/instances_train2017.json
json2_path = ./annotations/instances_val2017.json
save_path = ./instances_trainval2017.json
merge_keys = ['images', 'annotations']

-----------------------------------------------Merge------------------------------------------------

json read...

json merge...
info
licenses
images merge!
annotations merge!
categories

json save...

finish!
```
