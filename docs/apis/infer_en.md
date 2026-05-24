[简体中文](infer_cn.md) | English

# PaddleRS Inference APIs

The dynamic graph inference and static graph inference of PaddleRS are provided by the trainer ([`BaseModel`](https://github.com/PaddlePaddle/PaddleRS/blob/develop/paddlers/tasks/base.py) and its subclasses) and the **predictor** (`paddlers.deploy.Predictor`), respectively.

## Dynamic Graph Inference APIs

### Whole Image Inference

#### `BaseChangeDetector.predict()`

Interface:

```python
def predict(self, img_file, transforms=None):
```

Input parameters:

|Parameter Name|Type|Parameter Description|Default Value|
|-------|----|--------|-----|
|`img_file`|`list[tuple]` \| `tuple[str \| numpy.ndarray]`|Input image pair data (in NumPy array form) or input image pair paths. If there is only one image pair, a tuple should be used to sequentially contain the first and the second images (data of paths). If a group of image pairs need to be processed, a list that contains the data or paths of those image pairs (one tuple for each image pair) should be used.||
|`transforms`|`paddlers.transforms.Compose` \| `None`|Apply data transformation operators to input data. If `None`, the data transformation operators of trainer in the validation phase is used.|`None`|

Return format:

If `img_file` is a tuple, returns a dictionary containing the following key-value pairs:

```
{"label_map": category labels of model output (arranged in [h, w] format), "score_map": class probabilities of model output (arranged in format [h, w, c])}
```

If `img_file` is a list, returns an list as long as `img_file`, where each item is a dictionary (key-value pairs shown above), corresponding in order to each element in `img_file`.

#### `BaseClassifier.predict()`

Interface:

```python
def predict(self, img_file, transforms=None):
```

Input parameters:

|Parameter Name|Type|Parameter Description|Default Value|
|-------|----|--------|-----|
|`img_file`|`list[str\|numpy.ndarray]` \| `str` \| `numpy.ndarray`|input image data (in the form of NumPy array) or input image path. If a group of images need to be processed, the list should contain the data or paths for those images (one element for each image).||
|`transforms`|`paddlers.transforms.Compose` \| `None`|Apply data transformation operators to input data. If `None`, the data transformation operators of trainer in the validation phase is used.|`None`|

Return format:

If `img_file` is a string or NumPy array, returns a dictionary containing the following key-value pairs:

```
{"class_ids_map": output category IDs,
 "scores_map": output category probabilities,
 "label_names_map": output category names}
```

If `img_file` is a list, returns a list as long as `img_file`, where each item is a dictionary (key-value pairs shown above), corresponding in order to each element in `img_file`.

#### `BaseDetector.predict()`

Interface:

```python
def predict(self, img_file, transforms=None):
```

Input parameters:

|Parameter Name|Type|Parameter Description|Default Value|
|-------|----|--------|-----|
|`img_file`|`list[str\|numpy.ndarray]` \| `str` \| `numpy.ndarray`|input image data (in the form of NumPy array) or input image path. If a group of images need to be processed, the list should contain the data or paths for those images (one element for each image).||
|`transforms`|`paddlers.transforms.Compose` \| `None`|Apply data transformation operators to input data. If `None`, the data transformation operators of trainer in the validation phase is used.|`None`|

Return format:

If `img_file` is a string or NumPy array, returns a list with a predicted target box for each element in the list. The elements in the list are dictionaries containing the following key-value pairs:

```
{"category_id": Category ID,
 "category": Category name,
 "bbox": Bounding box position information, including the horizontal and vertical coordinates of the upper left corner of the box and the width and height of the box (for horizontal bounding boxes), or the horizontal and vertical coordinates of the four corners of the box (for rotated bounding boxes),  
 "score": Category confidence score,
 "mask": [RLE Format](https://baike.baidu.com/item/rle/366352) mask, only instance segmentation model prediction results contain this key-value pair}
```

If `img_file` is a list, returns a list as long as `img_file`, where each item is a list of dictionaries (key-value pairs shown above), corresponding in order to each element in `img_file`.

#### `BaseRestorer.predict()`

Interface:

```python
def predict(self, img_file, transforms=None):
```

Input parameters:

|Parameter Name|Type|Parameter Description|Default Value|
|-------|----|--------|-----|
|`img_file`|`list[str\|numpy.ndarray]` \| `str` \| `numpy.ndarray`|input image data (in the form of NumPy array) or input image path. If a group of images need to be processed, the list should contain the data or paths for those images (one element for each image).||
|`transforms`|`paddlers.transforms.Compose` \| `None`|Apply data transformation operators to input data. If `None`, the data transformation operators of trainer in the validation phase is used.|`None`|

Return format:

If `img_file` is a string or NumPy array, returns a dictionary containing the following key-value pairs:

```
{"res_map": restored or reconstructed images (arranged in [h, w, c] format)}
```

If `img_file` is a list, returns a list as long as `img_file`, where each item is a dictionary (key-value pairs shown above), corresponding in order to each element in `img_file`.

#### `BaseSegmenter.predict()`

Interface:

```python
def predict(self, img_file, transforms=None):
```

Input parameters:

|Parameter Name|Type|Parameter Description|Default Value|
|-------|----|--------|-----|
|`img_file`|`list[str\|numpy.ndarray]` \| `str` \| `numpy.ndarray`|input image data (in the form of NumPy array) or input image path. If a group of images need to be processed, the list should contain the data or paths for those images (one element for each image).||
|`transforms`|`paddlers.transforms.Compose` \| `None`|Apply data transformation operators to input data. If `None`, the data transformation operators of trainer in the validation phase is used.|`None`|

Return format:

If `img_file` is a string or NumPy array, returns a dictionary containing the following key-value pairs:

```
{"label_map": predicted category labels (arranged in [h, w] format), "score_map": category probabilities of model output (arranged in [h, w, c] format)}
```

If `img_file` is a list, returns a list as long as `img_file`, where each item is a dictionary (key-value pairs shown above), corresponding in order to each element in `img_file`.

### Sliding Window Inference

Considering the large-scale nature of remote sensing image, PaddleRS provides sliding window inference support for some tasks. The sliding window inference feature of PaddleRS has the following characteristics:

1. In order to solve the problem of insufficient memory caused by reading the whole large image at once, PaddleRS has adopted the lazy loading technology, which only read and process the image block in one window at a time.
2. Users can customize the size and stride of the sliding window. Meanwhile, PaddleRS allows overlapped sliding windows. For the overlapping parts between windows, PaddleRS will automatically fuse the predicted results.
3. The inference results can be saved in GeoTiff format, and the reading and writing of geographic transformation information and geographic projection information is supported.

Currently, the image segmentation trainer ([`BaseSegmenter`](https://github.com/PaddlePaddle/PaddleRS/blob/develop/paddlers/tasks/segmenter.py) and its subclasses) and change detection trainer ([`BaseChangeDetector`](https://github.com/PaddlePaddle/PaddleRS/blob/develop/paddlers/tasks/change_detector.py) and its subclasses) have dynamic graph sliding window inference APIs. Take the API of image segmentation trainer as an example, the explanation is as follows:

Interface:

```python
def slider_predict(self,
                   img_file,
                   save_dir,
                   block_size,
                   overlap=36,
                   transforms=None,
                   invalid_value=255,
                   merge_strategy='keep_last',
                   batch_size=1,
                   eager_load=False,
                   quiet=False):
```

Input parameter list:

|Parameter Name|Type|Parameter Description|Default Value|
|-------|----|--------|-----|
|`img_file`|`str`|Input image path.||
|`save_dir`|`str`|Directory to store predicted results.||
|`block_size`|`list[int]` \| `tuple[int]` \| `int`|Size of the sliding window. Specify the width and height in a list or tuple, or the same width and height in an integer.||
|`overlap`|`list[int]` \| `tuple[int]` \| `int`|Overlap of adjacent windows in pixels. Specify the width and height in a list or tuple, or the same width and height in an integer).|`36`|
|`transforms`|`paddlers.transforms.Compose` \| `None`|Apply data transformation operators to input data. If `None`, the data transformation operators of trainer in the validation phase is used.|`None`|
|`invalid_value`|`int`|Value used to mark invalid pixels in the output image.|`255`|
|`merge_strategy`|`str`|Strategies used to merge sliding window overlapping areas.`'keep_first'` means retaining the prediction category of the first window in the traversal order (left to right, top to bottom, column first); `'keep_last'` stands for keeping the prediction category of the last window in the traversal order; `'accum'` means determining the final prediction categories in the overlapping area by summing the prediction probabilities given by each window. It should be noted that when dense inference with large `overlap` is carried out for large size images, using `'accum'` strategy may lead to longer inference time, but generally it can achieve better performance at the window boundary.|`'keep_last'`|
|`batch_size`|`int`|Mini-batch size used for prediction.|`1`|
|`eager_load`|`bool`|If `True`, instead of using lazy loading, the entire image is loaded into memory at the beginning of the prediction.|`False`|
|`quiet`|`bool`|If `True`, the predicted progress will not be displayed.|`False`|

The sliding window inference API of the change detection trainer is similar to that of the image segmentation trainer, but it should be noted that the information stored in the output results, such as geographic transformation and projection, is subject to the information read from the first temporal image, and the output file name is the same as that of the first temporal image file.

## Static Graph Inference APIs

### Python API

First, [export the model to a deployment format](https://github.com/PaddlePaddle/PaddleRS/blob/develop/deploy/export/README.md) or execute model quantization. Then, PaddleRS provides `paddlers.deploy.Predictor` to load the deployment model or quantization model and perform inference based on the [Paddle Inference](https://www.paddlepaddle.org.cn/tutorials/projectdetail/3952715) engine.

#### Initialize `Predictor`

`Predictor.__init__()` takes the following arguments:

|Parameter Name|Type|Parameter Description|Default Value|
|-------|----|--------|-----|
|`model_dir`|`str`|Model path (must be an exported model).||
|`use_gpu`|`bool`|Whether to use GPU.|`False`|
|`gpu_id`|`int`|ID of the GPU used.|`0`|
|`cpu_thread_num`|`int`|Number of threads when inference is performed using CPUs.|`1`|
|`use_mkl`|`bool`|Whether to use MCL-DNN library. This option takes effect only when inference is performed using CPUs.|`False`|
|`mkl_thread_num`|`int`|Count the threads of MKL-DNN.|`4`|
|`use_trt`|`bool`|Whether to use TensorRT.|`False`|
|`use_glog`|`bool`|Whether to enable glog logs.|`False`|
|`memory_optimize`|`bool`|Whether to enable memory optimization.|`True`|
|`max_trt_batch_size`|`int`|Maximum batch size configured when TensorRT is used.|`1`|
|`trt_precision_mode`|`str`|Precision to be used when using TensorRT, with the optional values of `'float32'` or `'float16'`.|`'float32'`|

#### `Predictor.predict()`

Interface:

```python
def predict(self,
            img_file,
            topk=1,
            transforms=None,
            warmup_iters=0,
            repeats=1):
```

Input parameter list:

|Parameter Name|Type|Parameter Description|Default Value|
|-------|----|--------|-----|
|`img_file`|`list[str\|tuple\|numpy.ndarray]` \| `str` \| `tuple` \| `numpy.ndarray`|For scene classification, object detection, image restoration and image segmentation tasks, this parameter can be a single image path, decoded image data in [h, w, c] format with a float32 type (expressed as NumPy array), or a list of image paths or `numpy.ndarray` objects. For the change detection task, this parameter can be a two-tuple of image paths, a two-tuple composed of two decoded images, or a list composed of one of the above two kinds of tuples.||
|`topk`|`int`|It is used in scene classification model prediction, indicating that the categories with the top `topk` output probability should be selected as the final results.|`1`|
|`transforms`|`paddlers.transforms.Compose`\|`None`|Apply data transformation operators to input data. If `None`, the operators read from `model.yml` is used.|`None`|
|`warmup_iters`|`int`|Number of warm-up rounds used to evaluate model inference and pre- and post-processing speed. If it is greater than `1`, the inference will first be repeated for `warmup_iters` iterations, before the speed is accessed.|`0`|
|`repeats`|`int`|Number of repetitions used to assess model reasoning and pre- and post-processing speed. If it is greater than `1`, repeats the prediction and averages the time.|`1`|
|`quiet`|`bool`|If `True`, no timing information will be printed.|`False`|

`Predictor.predict()`returns exactly the same format as the graph inference API. For details, please refer to [Dynamic Graph Inference APIs](#dynamic-graph-inference-apis).

### `Predictor.slider_predict()`

Implements the sliding window inference function. It is used in the same way as `slider_predict()` of `BaseSegmenter` and `BaseChangeDetector`.
