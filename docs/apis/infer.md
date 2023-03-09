# PaddleRS Inference API illustration

PaddleRS and the static and dynamic figure reasoning figure reasoning respectively by the trainer ([` BaseModel `] (https://github.com/PaddlePaddle/PaddleRS/blob/develop/paddlers/tasks/base.py) And its subclasses) And **Predictor** (`paddlers.deploy.predictor`) to provide.

## Dynamic graph inference API

### Whole graph inference

#### `BaseChangeDetector.predict()`

Interface form：

```python
def predict(self, img_file, transforms=None):
```

Input parameters:

|parameter names|type|parameter description|default|
|-------|----|--------|-----|
|`img_file`|`list[tuple]` \| `tuple[str\|np.ndarray]`|Input image pair data (in NumPy array form) or input image pair path.If only one image pair is predicted, a tuple is used to sequentially contain the first phase image data/path and the second phase image data/path. 

If you need to predict a group of image pairs at once, the list contains the data or paths of those image pairs (one tuple from the list for each image pair).||

|`transforms`|`paddlers.transforms.Compose` \| `None`|There is a data transformation operator applied to input data. If it is 'None', then the data transformation operator used by the trainer during the validation phase is used.|`None`|

Return format：

If 'img_file' is a tuple, the returned object is a dictionary with the following key-value pairs:

```
{"label_map": output class labels (in [h, w] format), "score_map": the class probabilities of the model output (in [h, w, c] format)}
```

If 'img_file' is a list, then the returned object is a list of the same length as' img_file ', where each item is a dictionary (key/value pair as shown above), in the order corresponding to each item in 'img_file'.

#### `BaseClassifier.predict()`

Interface form：

```python
def predict(self, img_file, transforms=None):
```

Input parameters:

|parameter names|type|parameter description|default|
|-------|----|--------|-----|
|`img_file`|`list[str\|np.ndarray]` \| `str` \| `np.ndarray`|Input image data (NumPy array) or input image path. If we need to predict a group of images at once, we can create a list that contains the data or paths for the images (one element for each image).||
|`transforms`|`paddlers.transforms.Compose` \| `None`|There are data transformation operators applied to the input data.If it is' None ', then the data transformation operator used by the trainer during the validation phase is used.|`None`|

Return format：

If'img_file'is a string or NumPy array, the returned object is a dictionary with the following key-value pairs:

```
{"class_ids_map": Output class labels,
 "scores_map": Output class probability,
 "label_names_map": Output class name}
```

If 'img_file' is a list, then the returned object is a list of the same length as' img_file ', where each item is a dictionary (key/value pair as shown above), in the order corresponding to each item in 'img_file'.

#### `BaseDetector.predict()`

Interface form：

```python
def predict(self, img_file, transforms=None):
```

Input parameters:

|parameter names|type|parameter description|default|
|-------|----|--------|-----|
|`img_file`|`list[str\|np.ndarray]` \| `str` \| `np.ndarray`|Enter the image data (in the form of NumPy array) or enter the image path. If you need to predict a group of images at once, the list contains the data or paths for those images (one element in the list for each image).||
|`transforms`|`paddlers.transforms.Compose` \| `None`|There are data transformation operators applied to the input data.If it is' None ', then the data transformation operator used by the trainer during the validation phase is used.|`None`|

Return format：

If 'img_file' is a string or NumPy array, the returned object is a list with one element for each predicted target box. The elements in the list will be dictionaries containing the following key-value pairs:

```
{"category_id": class ID,
 "category": class name,
 "bbox": The target box position information, in turn, contains the horizontal and vertical coordinates of the top left corner of the target box, and the width and length of the target box.
 "score": Class confidence,
 "mask": [RLE format](https://baike.baidu.com/item/rle/366352)'s mask graph，Only instance segmentation model prediction results contain this key-value pair}
```

If 'img_file' is a list, then the returned object is a list of the same length as'img_file', where each item is a list of dictionaries (key/value pairs as shown above), in the order corresponding to each item in 'img_file'.

#### `BaseRestorer.predict()`

Interface form：

```python
def predict(self, img_file, transforms=None):
```

Input parameters:

|parameter names|type|parameter description|default|
|-------|----|--------|-----|
|`img_file`|`list[str\|np.ndarray]` \| `str` \| `np.ndarray`|Input image data (NumPy array) or input image path. If we need to predict a group of images at once, we can create a list that contains the data or paths for the images (one element for each image).||
|`transforms`|`paddlers.transforms.Compose` \| `None`|There are data transformation operators applied to the input data.If it is' None ', then the data transformation operator used by the trainer during the validation phase is used.|`None`|

Return format：

If 'img_file' is a string or NumPy array, the returned object is a dictionary with the following key-value pairs:
```
{"res_map": Restored or reconstructed images (arranged in [h, w, c] format) output by the model）}
```

If 'img_file' is a list, then the returned object is a list of the same length as' img_file ', where each item is a dictionary (key/value pair as shown above), in the order corresponding to each item in 'img_file'.

#### `BaseSegmenter.predict()`

Interface form：

```python
def predict(self, img_file, transforms=None):
```

Input parameters:

|parameter names|type|parameter description|default|
|-------|----|--------|-----|
|`img_file`|`list[str\|np.ndarray]` \| `str` \| `np.ndarray`|Input image data (NumPy array) or input image path. If we need to predict a group of images at once, we can create a list that contains the data or paths for the images (one element for each image).||
|`transforms`|`paddlers.transforms.Compose` \| `None`|There are data transformation operators applied to the input data.If it is' None ', then the data transformation operator used by the trainer during the validation phase is used.|`None`|

Return format：

If 'img_file' is a string or NumPy array, the returned object is a dictionary with the following key-value pairs:

```
{"label_map": Output class labels(in [h, w] format), "score_map": The class probabilities of the model output (in [h, w, c] format)}
```

If 'img_file' is a list, then the returned object is a list of the same length as' img_file ', where each item is a dictionary (key/value pair as shown above), in the order corresponding to each item in 'img_file'.
### Sliding window reasoning

Considering the large format nature of remote sensing images, PaddleRS provides sliding window inference support for some tasks. PaddleRS 'sliding window reasoning features include:
1. To solve the problem of memory shortage caused by reading the entire detail image at once, PaddleRS uses lazy memory loading technology, which only reads and processes image blocks in one window at a time.
2. The user can customize the size and step of the sliding window. Sliding window overlap is supported, and PaddleRS will automatically fuse the model prediction results for the overlapping parts between Windows.
3. It supports saving the reasoning results to GeoTiff format, and supports reading and writing geographic transformation information and geographic projection information.

At present, the image segmentation trainers：（[`BaseSegmenter`](https://github.com/PaddlePaddle/PaddleRS/blob/develop/paddlers/tasks/segmenter.py) and its subclasses) with change detection trainers（[`BaseChangeDetector`](https://github.com/PaddlePaddle/PaddleRS/blob/develop/paddlers/tasks/change_detector.py) and its subclasses) have dynamic graph sliding window reasoning API, taking the API of image segmentation task as an example, the description is as follows:

Interface form：

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

Enter a list of parameters:

|parameter names|type|parameter description|default|
|-------|----|--------|-----|
|`img_file`|`str`|Input image path.||
|`save_dir`|`str`|The output path of the prediction result||
|`block_size`|`list[int]` \| `tuple[int]` \| `int`|The window size of the sliding window（Specifies width and height as a list or tuple, or the same width and height as an integer）||
|`overlap`|`list[int]` \| `tuple[int]` \| `int`|The sliding step of the window (width and height specified as a list or tuple, or the same width and height as an integer)|`36`|
|`transforms`|`paddlers.transforms.Compose` \| `None`|There are data transformation operators applied to the input data.If it is' None ', then the data transformation operator used by the trainer during the validation phase is used.|`None`|
|`invalid_value`|`int`|The value used to mark invalid pixels in the output image.|`255`|
|`merge_strategy`|`str`|Which is the strategy used to merge the overlapping regions of sliding Windows. "'keep_first' 'means keep the predicted class of the window with the highest traversal order (left to right, top to bottom, column first);
'keep_last' keeps the predicted class of the window with the lowest traversal order; 'accum' means that the final prediction class is calculated by summing the predicted probabilities given by each window in the overlapping region.
It should be noted that when performing dense inference on large images with large 'overlap', using the 'accum' 'strategy may lead to longer inference time, but generally achieves better performance on the window boundary.|`'keep_last'`|
|`batch_size`|`int`|The mini-batch size to use for prediction.|`1`|
|`eager_load`|`bool`|If it is' True ', then instead of using lazy memory loading, the whole image is loaded into memory at the beginning of prediction.|`False`|
|`quiet`|`bool`|If 'True', no progress will be shown.|`False`|

The sliding window inference API for the change detection task is similar to the image segmentation task, but it should be noted that the information stored in the output is based on the information read from the first phase image, and the file name of the sliding window inference result is the same as the first phase image file.

## Static graph reasoning API

### Python API

[Export models to deployment format](https://github.com/PaddlePaddle/PaddleRS/blob/develop/deploy/export/README.md)Or perform model quantization,PaddleRS provides 'Paddlers.deploy.predictor' for loading deployment or quantization format models and executing based on [Paddle Inference]] Reasoning (https://www.paddlepaddle.org.cn/tutorials/projectdetail/3952715).

#### Initializes the 'Predictor' object

`Predictor.__init__()` takes the following arguments：

|parameter names|type|parameter description|default|
|-------|----|--------|-----|
|`model_dir`|`str`|Model path (must be the exported deployment or quantization model).||
|`use_gpu`|`bool`|Weather to use GPU|`False`|
|`gpu_id`|`int`|Weather to use the ID of the GPU.|`0`|
|`cpu_thread_num`|`int`|Number of threads when performing inference using the CPU.|`1`|
|`use_mkl`|`bool`|Whether to use the MKL-DNN computation library (this option only works if the inference is performed using the CPU).|`False`|
|`mkl_thread_num`|`int`|Use MKL-DNN to counts the number of threads.|`4`|
|`use_trt`|`bool`|Whether to use TensorRT.|`False`|
|`use_glog`|`bool`|Whether glog logging is enabled.|`False`|
|`memory_optimize`|`bool`| whether to enable memory optimization.|`True`|
|`max_trt_batch_size`|`int`|Maximum batch size configured when using TensorRT.|`1`|
|`trt_precision_mode`|`str`|Precision to use with TensorRT, optionally "float32' 'or" float16''.|`'float32'`|

#### `Predictor.predict()`

Interface form：

```python
def predict(self,
            img_file,
            topk=1,
            transforms=None,
            warmup_iters=0,
            repeats=1):
```

Enter a list of parameters:
|parameter names|type|parameter description|default|
|-------|----|--------|-----|
|`img_file`|`list[str\|tuple\|np.ndarray]` \| `str` \| `tuple` \| `np.ndarray`|For scene classification, object detection, image restoration, and image segmentation tasks, this can be a single image path or the decoded [h, w, c] float32type image data (represented as a NumPy array).
Or a list of image paths or np.ndarray objects. For change detection, this can be a tuple of image paths (one for each temporal path), a tuple of two decoded images, or a list of one of these two tuples.||
|`topk`|`int`|When used in scene classification model prediction, it means that the top 'topk' categories of model output probability are selected as the final result.|`1`|
|`transforms`|`paddlers.transforms.Compose`\|`None`|There are data transformation operators applied to the input data.For 'None', the operator read from 'model.yml' is used.|`None`|
|`warmup_iters`|`int`|Number of warm-up rounds to evaluate model inference as well as pre and post processing speed. If it is greater than 1, the 'warmup_iters' inference is repeated before the formal prediction and speed evaluation begins.|`0`|
|`repeats`|`int`|Number of repetitions to evaluate model inference and pre - and post-processing speed. If it is greater than 1, 'repeats' are performed and the time average is taken.|`1`|
|`quiet`|`bool`|If 'True', no timing information will be printed.|`False`|

`Predictor.predict()`Its return format is exactly the same as that of the corresponding graph inference API. For details, see Graph Inference API (# Graph Inference api).

### `Predictor.slider_predict()`

It can realize sliding window reasoning function. The usage is the same as the 'slider_predict()' method of 'BaseSegmenter' and 'BaseChangeDetector'.
