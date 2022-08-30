# PaddleRS推理API说明

PaddleRS的动态图推理和静态图推理能力分别由训练器（[`BaseModel`](https://github.com/PaddlePaddle/PaddleRS/blob/develop/paddlers/tasks/base.py)及其子类）和**预测器**（`paddlers.deploy.Predictor`）提供。

## 动态图推理API

### 整图推理

#### `BaseChangeDetector.predict()`

接口形式：

```python
def predict(self, img_file, transforms=None):
```

输入参数：

|参数名称|类型|参数说明|默认值|
|-------|----|--------|-----|
|`img_file`|`list[tuple]` \| `tuple[str\|np.ndarray]`|输入影像对数据（NumPy数组形式）或输入影像对路径。若仅预测一个影像对，使用一个元组顺序包含第一时相影像数据/路径以及第二时相影像数据/路径。若需要一次性预测一组影像对，以列表包含这些影像对的数据或路径（每个影像对对应列表中的一个元组）。||
|`transforms`|`paddlers.transforms.Compose` \| `None`|对输入数据应用的数据变换算子。若为`None`，则使用训练器在验证阶段使用的数据变换算子。|`None`|

返回格式：

若`img_file`是一个元组，则返回对象为包含下列键值对的字典：

```
{"label_map": 输出类别标签（以[h, w]格式排布），"score_map": 模型输出的各类别概率（以[h, w, c]格式排布）}
```

若`img_file`是一个列表，则返回对象为与`img_file`等长的列表，其中的每一项为一个字典（键值对如上所示），顺序对应`img_file`中的每个元素。

#### `BaseClassifier.predict()`

接口形式：

```python
def predict(self, img_file, transforms=None):
```

输入参数：

|参数名称|类型|参数说明|默认值|
|-------|----|--------|-----|
|`img_file`|`list[str\|np.ndarray]` \| `str` \| `np.ndarray`|输入影像数据（NumPy数组形式）或输入影像路径。若需要一次性预测一组影像，以列表包含这些影像的数据或路径（每幅影像对应列表中的一个元素）。||
|`transforms`|`paddlers.transforms.Compose` \| `None`|对输入数据应用的数据变换算子。若为`None`，则使用训练器在验证阶段使用的数据变换算子。|`None`|

返回格式：

若`img_file`是一个字符串或NumPy数组，则返回对象为包含下列键值对的字典：

```
{"label_map": 输出类别标签,
 "scores_map": 输出类别概率,
 "label_names_map": 输出类别名称}
```

若`img_file`是一个列表，则返回对象为与`img_file`等长的列表，其中的每一项为一个字典（键值对如上所示），顺序对应`img_file`中的每个元素。

#### `BaseDetector.predict()`

接口形式：

```python
def predict(self, img_file, transforms=None):
```

输入参数：

|参数名称|类型|参数说明|默认值|
|-------|----|--------|-----|
|`img_file`|`list[str\|np.ndarray]` \| `str` \| `np.ndarray`|输入影像数据（NumPy数组形式）或输入影像路径。若需要一次性预测一组影像，以列表包含这些影像的数据或路径（每幅影像对应列表中的一个元素）。||
|`transforms`|`paddlers.transforms.Compose` \| `None`|对输入数据应用的数据变换算子。若为`None`，则使用训练器在验证阶段使用的数据变换算子。|`None`|

返回格式：

若`img_file`是一个字符串或NumPy数组，则返回对象为一个列表，列表中每个元素对应一个预测的目标框。列表中的元素为包含下列键值对的字典：

```
{"category_id": 类别ID,
 "category": 类别名称,
 "bbox": 目标框位置信息，依次包含目标框左上角的横、纵坐标以及目标框的宽度和长度,  
 "score": 类别置信度,
 "mask": [RLE格式](https://baike.baidu.com/item/rle/366352)的掩模图（mask），仅实例分割模型预测结果包含此键值对}
```

若`img_file`是一个列表，则返回对象为与`img_file`等长的列表，其中的每一项为一个由字典（键值对如上所示）构成的列表，顺序对应`img_file`中的每个元素。

#### `BaseRestorer.predict()`



#### `BaseSegmenter.predict()`

接口形式：

```python
def predict(self, img_file, transforms=None):
```

输入参数：

|参数名称|类型|参数说明|默认值|
|-------|----|--------|-----|
|`img_file`|`list[str\|np.ndarray]` \| `str` \| `np.ndarray`|输入影像数据（NumPy数组形式）或输入影像路径。若需要一次性预测一组影像，以列表包含这些影像的数据或路径（每幅影像对应列表中的一个元素）。||
|`transforms`|`paddlers.transforms.Compose` \| `None`|对输入数据应用的数据变换算子。若为`None`，则使用训练器在验证阶段使用的数据变换算子。|`None`|

返回格式：

若`img_file`是一个字符串或NumPy数组，则返回对象为包含下列键值对的字典：

```
{"label_map": 输出类别标签（以[h, w]格式排布），"score_map": 模型输出的各类别概率（以[h, w, c]格式排布）}
```

若`img_file`是一个列表，则返回对象为与`img_file`等长的列表，其中的每一项为一个字典（键值对如上所示），顺序对应`img_file`中的每个元素。

### 滑窗推理

考虑到遥感影像的大幅面性质，PaddleRS为部分任务提供了滑窗推理支持。PaddleRS的滑窗推理功能具有如下特色：

1. 为了解决一次读入整张大图直接导致内存不足的问题，PaddleRS采用延迟载入内存的技术，一次仅读取并处理一个窗口内的影像块。
2. 用户可自定义滑窗的大小和步长。支持滑窗重叠，对于窗口之间重叠的部分，PaddleRS将自动对模型预测结果进行融合。
3. 支持将推理结果保存为GeoTiff格式，支持对地理变换信息、地理投影信息的读取与写入。

目前，图像分割训练器（[`BaseSegmenter`](https://github.com/PaddlePaddle/PaddleRS/blob/develop/paddlers/tasks/segmenter.py)及其子类）与变化检测训练器（[`BaseChangeDetector`](https://github.com/PaddlePaddle/PaddleRS/blob/develop/paddlers/tasks/change_detector.py)及其子类）具有动态图滑窗推理API，以图像分割任务的API为例，说明如下：

接口形式：

```python
def slider_predict(self,
                   img_file,
                   save_dir,
                   block_size,
                   overlap=36,
                   transforms=None):
```

输入参数列表：

|参数名称|类型|参数说明|默认值|
|-------|----|--------|-----|
|`img_file`|`str`|输入影像路径。||
|`save_dir`|`str`|预测结果输出路径。||
|`block_size`|`list[int]` \| `tuple[int]` \| `int`|滑窗的窗口大小（以列表或元组指定长、宽或以一个整数指定相同的长宽）。||
|`overlap`|`list[int]` \| `tuple[int]` \| `int`|滑窗的滑动步长（以列表或元组指定长、宽或以一个整数指定相同的长宽）。|`36`|
|`transforms`|`paddlers.transforms.Compose` \| `None`|对输入数据应用的数据变换算子。若为`None`，则使用训练器在验证阶段使用的数据变换算子。|`None`|

变化检测任务的滑窗推理API与图像分割任务类似，但需要注意的是输出结果中存储的地理变换、投影等信息以从第一时相影像中读取的信息为准。

## 静态图推理API

### Python API

[将模型导出为部署格式](https://github.com/PaddlePaddle/PaddleRS/blob/develop/deploy/export/README.md)或执行模型量化后，PaddleRS提供`paddlers.deploy.Predictor`用于加载部署或量化格式模型以及执行基于[Paddle Inference](https://www.paddlepaddle.org.cn/tutorials/projectdetail/3952715)的推理。

#### 初始化`Predictor`对象

`Predictor.__init__()`接受如下参数：

|参数名称|类型|参数说明|默认值|
|-------|----|--------|-----|
|`model_dir`|`str`|模型路径（必须是导出的部署或量化模型）。||
|`use_gpu`|`bool`|是否使用GPU。|`False`|
|`gpu_id`|`int`|使用GPU的ID。|`0`|
|`cpu_thread_num`|`int`|使用CPU执行推理时的线程数。|`1`|
|`use_mkl`|`bool`|是否使用MKL-DNN计算库（此选项仅在使用CPU执行推理时生效）。|`False`|
|`mkl_thread_num`|`int`|MKL-DNN计算线程数。|`4`|
|`use_trt`|`bool`|是否使用TensorRT。|`False`|
|`use_glog`|`bool`|是否启用glog日志。|`False`|
|`memory_optimize`|`bool`|是否启用内存优化。|`True`|
|`max_trt_batch_size`|`int`|在使用TensorRT时配置的最大batch size。|`1`|
|`trt_precision_mode`|`str`|在使用TensorRT时采用的精度，可选值为`'float32'`或`'float16'`。|`'float32'`|

#### `Predictor.predict()`

接口形式：

```python
def predict(self,
            img_file,
            topk=1,
            transforms=None,
            warmup_iters=0,
            repeats=1):
```

输入参数列表：

|参数名称|类型|参数说明|默认值|
|-------|----|--------|-----|
|`img_file`|`list[str\|tuple\|np.ndarray]` \| `str` \| `tuple` \| `np.ndarray`|对于场景分类、目标检测和图像分割任务来说，该参数可为单一图像路径，或是解码后的、排列格式为[h, w, c]且具有float32类型的图像数据（表示为NumPy数组形式），或者是一组图像路径或np.ndarray对象构成的列表；对于变化检测任务来说，该参数可以为图像路径二元组（分别表示前后两个时相影像路径），或是解码后的两幅图像组成的二元组，或者是上述两种二元组之一构成的列表。||
|`topk`|`int`|场景分类模型预测时使用，表示选取模型输出概率大小排名前`topk`的类别作为最终结果。|`1`|
|`transforms`|`paddlers.transforms.Compose`\|`None`|对输入数据应用的数据变换算子。若为`None`，则使用从`model.yml`中读取的算子。|`None`|
|`warmup_iters`|`int`|预热轮数，用于评估模型推理以及前后处理速度。若大于1，将预先重复执行`warmup_iters`次推理，而后才开始正式的预测及其速度评估。|`0`|
|`repeats`|`int`|重复次数，用于评估模型推理以及前后处理速度。若大于1，将执行`repeats`次预测并取时间平均值。|`1`|

`Predictor.predict()`的返回格式与相应的动态图推理API的返回格式完全相同，详情请参考[动态图推理API](#动态图推理api)。
