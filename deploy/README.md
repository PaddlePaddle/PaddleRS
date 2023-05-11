# Python部署

PaddleRS已经集成了基于Python的高性能预测（prediction）接口。在安装PaddleRS后，可参照如下代码示例执行预测。

## 1 部署模型导出

在服务端部署模型时需要首先将训练过程中保存的模型导出为部署格式，具体的导出步骤请参考文档[部署模型导出](export/README.md)。

## 2 预测接口调用

使用预测接口的基本流程为：首先构建`Predictor`对象，然后调用`Predictor`的`predict()`方法执行预测。需要说明的是，`Predictor`对象的`predict()`方法返回的结果与对应的训练器（在`paddlers/tasks/`目录的文件中定义）的`predict()`方法返回结果具有相同的格式。

### 2.1 基本使用

以变化检测任务为例，说明预测接口的基本使用方法：

```python
from paddlers.deploy import Predictor

# 第一步：构建Predictor。该类接受的构造参数如下：
#     model_dir: 模型路径（必须是导出的部署或量化模型）。
#     use_gpu: 是否使用GPU，默认为False。
#     gpu_id: 使用GPU的ID，默认为0。
#     cpu_thread_num：使用CPU进行预测时的线程数，默认为1。
#     use_mkl: 是否使用MKL-DNN计算库，CPU情况下使用，默认为False。
#     mkl_thread_num: MKL-DNN计算线程数，默认为4。
#     use_trt: 是否使用TensorRT，默认为False。
#     use_glog: 是否启用glog日志, 默认为False。
#     memory_optimize: 是否启动内存优化，默认为True。
#     max_trt_batch_size: 在使用TensorRT时配置的最大batch size，默认为1。
#     trt_precision_mode：在使用TensorRT时采用的精度，可选值['float32', 'float16']。默认为'float32'。
#
# 下面的语句构建的Predictor对象依赖static_models/目录中存储的部署格式模型，并使用GPU进行推理。
predictor = Predictor("static_models/", use_gpu=True)

# 第二步：调用Predictor的predict()方法执行推理。该方法接受的输入参数如下：
#     img_file: 对于场景分类、图像复原、目标检测和图像分割任务来说，该参数可为单一图像路径，或是解码后的、排列格式为（H, W, C）
#         且具有float32类型的图像数据（表示为numpy的ndarray形式），或者是一组图像路径或np.ndarray对象构成的列表；对于变化检测
#         任务来说，该参数可以为图像路径二元组（分别表示前后两个时相影像路径），或是两幅图像组成的二元组，或者是上述两种二元组
#         之一构成的列表。
#     topk: 场景分类模型预测时使用，表示选取模型输出概率大小排名前`topk`的类别作为最终结果。默认值为1。
#     transforms: 对输入数据应用的数据变换算子。若为None，则使用从`model.yml`中读取的算子。默认值为None。
#     warmup_iters: 预热轮数，用于评估模型推理以及前后处理速度。若大于1，会预先重复执行`warmup_iters`次推理，而后才开始正式的预测及其速度评估。默认值为0。
#     repeats: 重复次数，用于评估模型推理以及前后处理速度。若大于1，会执行`repeats`次预测并取时间平均值。默认值为1。
#
# 下面的语句传入两幅输入影像的路径
res = predictor.predict(("demo_data/A.png", "demo_data/B.png"))

# 第三步：解析predict()方法返回的结果。
#     对于图像分割和变化检测任务而言，predict()方法返回的结果为一个字典或字典构成的列表。字典中的`label_map`键对应的值为类别标签图，对于二值变化检测
#     任务而言只有0（不变类）或者1（变化类）两种取值；`score_map`键对应的值为类别概率图，对于二值变化检测任务来说一般包含两个通道，第0个通道表示不发生
#     变化的概率，第1个通道表示发生变化的概率。如果返回的结果是由字典构成的列表，则列表中的第n项与输入的img_file中的第n项对应。
#
# 下面的语句从res中解析二值变化图（binary change map）
cm_1024x1024 = res['label_map']
```

请注意，**`predictor.predict()`方法接受的影像列表长度与导出模型时指定的batch size必须一致**（若指定的batch size不为-1），这是因为`Predictor`对象将所有输入影像拼接成一个batch执行预测。您可以在[模型推理API说明](../docs/apis/infer_cn.md)中了解关于`predictor.predict()`方法返回结果格式的更多信息。

### 2.2 指定预热轮数与重复次数

加载模型后，对前几张图片的预测速度会较慢，这是因为程序刚启动时需要进行内存、显存初始化等步骤。通常，在处理20-30张图片后，模型的预测速度能够达到稳定值。基于这一观察，**如果需要评估模型的预测速度，可通过指定预热轮数`warmup_iters`对模型进行预热**。此外，**为获得更加精准的预测速度估计值，可指定重复`repeats`次预测后计算平均耗时**。指定预热轮数与重复次数的一个简单例子如下：

```python
import paddlers as pdrs

predictor = pdrs.deploy.Predictor('./inference_model')
result = predictor.predict(img_file='test.jpg',
                           warmup_iters=100,
                           repeats=100)
```
