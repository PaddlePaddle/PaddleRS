# 使用教程——训练模型

本目录中整理了使用PaddleRS训练模型的示例代码。代码中均提供对示例数据的自动下载，并均使用GPU对模型进行训练。

|示例代码路径 | 任务 | 模型 |
|------|--------|---------|
|change_detection/bit.py | 变化检测 | BIT |
|change_detection/cdnet.py | 变化检测 | CDNet |
|change_detection/changeformer.py | 变化检测 | ChangeFormer |
|change_detection/dsamnet.py | 变化检测 | DSAMNet |
|change_detection/dsifn.py | 变化检测 | DSIFN |
|change_detection/fc_ef.py | 变化检测 | FC-EF |
|change_detection/fc_siam_conc.py | 变化检测 | FC-Siam-conc |
|change_detection/fc_siam_diff.py | 变化检测 | FC-Siam-diff |
|change_detection/fccdn.py | 变化检测 | FCCDN |
|change_detection/p2v.py | 变化检测 | P2V-CD |
|change_detection/snunet.py | 变化检测 | SNUNet |
|change_detection/stanet.py | 变化检测 | STANet |
|classification/condensenetv2.py | 场景分类 | CondenseNet V2 |
|classification/hrnet.py | 场景分类 | HRNet |
|classification/mobilenetv3.py | 场景分类 | MobileNetV3 |
|classification/resnet50_vd.py | 场景分类 | ResNet50-vd |
|image_restoration/drn.py | 图像复原 | DRN |
|image_restoration/esrgan.py | 图像复原 | ESRGAN |
|image_restoration/lesrcnn.py | 图像复原 | LESRCNN |
|object_detection/faster_rcnn.py | 目标检测 | Faster R-CNN |
|object_detection/ppyolo.py | 目标检测 | PP-YOLO |
|object_detection/ppyolo_tiny.py | 目标检测 | PP-YOLO Tiny |
|object_detection/ppyolov2.py | 目标检测 | PP-YOLOv2 |
|object_detection/yolov3.py | 目标检测 | YOLOv3 |
|semantic_segmentation/bisenetv2.py | 图像分割 | BiSeNet V2 |
|semantic_segmentation/deeplabv3p.py | 图像分割 | DeepLab V3+ |
|semantic_segmentation/factseg.py | 图像分割 | FactSeg |
|semantic_segmentation/farseg.py | 图像分割 | FarSeg |
|semantic_segmentation/fast_scnn.py | 图像分割 | Fast-SCNN |
|semantic_segmentation/hrnet.py | 图像分割 | HRNet |
|semantic_segmentation/unet.py | 图像分割 | UNet |

## 环境准备

+ [PaddlePaddle安装](https://www.paddlepaddle.org.cn/install/quick)
  - 版本要求：PaddlePaddle>=2.2.0

+ PaddleRS安装

PaddleRS代码会跟随开发进度不断更新，可以安装develop分支的代码使用最新的功能，安装方式如下：

```shell
git clone https://github.com/PaddlePaddle/PaddleRS
cd PaddleRS
git checkout develop
pip install -r requirements.txt
python setup.py install
```

若在使用`python setup.py install`时下载依赖缓慢或超时，可以在`setup.py`相同目录下新建`setup.cfg`，并输入以下内容，则可通过清华源进行加速下载：

```
[easy_install]
index-url=https://pypi.tuna.tsinghua.edu.cn/simple
```

+ （可选）GDAL安装

PaddleRS支持对多种类型卫星数据的读取。完整使用PaddleRS的遥感数据读取功能需要安装GDAL，安装方式如下：

  - Linux / MacOS

推荐使用conda进行安装:

```shell
conda install gdal
```

  - Windows

Windows用户可以在[此站点](https://www.lfd.uci.edu/~gohlke/pythonlibs/#gdal)下载与Python和系统版本相对应的.whl格式安装包到本地，以*GDAL‑3.3.3‑cp39‑cp39‑win_amd64.whl*为例，使用pip工具安装:

```shell
pip install GDAL‑3.3.3‑cp39‑cp39‑win_amd64.whl
```

### *Docker安装

1. 从dockerhub拉取:

```shell
docker pull paddlepaddle/paddlers:1.0.0  # 暂无
```

- （可选）从头开始构建，可以通过设置`PPTAG`选择PaddlePaddle的多种基础镜像，构建CPU或不同GPU环境:

```shell
git clone https://github.com/PaddlePaddle/PaddleRS
cd PaddleRS
docker build -t <imageName> .  # 默认为2.4.1的cpu版本
# docker build -t <imageName> . --build-arg PPTAG=2.4.1-gpu-cuda10.2-cudnn7.6-trt7.0  # 2.4.1的gpu版本之一
# 其余Tag可以参考：https://hub.docker.com/r/paddlepaddle/paddle/tags
```

2. 启动镜像

```shell
docker iamges  # 查看镜像的ID
docker run -it <imageID>
```

## 开始训练

+ 在安装完成PaddleRS后，使用如下命令执行单卡训练。脚本将自动下载训练数据。以DeepLab V3+图像分割模型为例：

```shell
# 指定需要使用的GPU设备编号
export CUDA_VISIBLE_DEVICES=0
python tutorials/train/semantic_segmentation/deeplabv3p.py
```

+ 如需使用多块GPU进行训练，例如使用2张显卡时，执行如下命令：

```shell
python -m paddle.distributed.launch --gpus 0,1 tutorials/train/semantic_segmentation/deeplabv3p.py
```

## VisualDL可视化训练指标

将传入`train()`方法的`use_vdl`参数设为`True`，则模型训练过程中将自动把训练日志以VisualDL的格式存储到`save_dir`（用户自己指定的路径）目录下名为`vdl_log`的子目录中。用户可以使用如下命令启动VisualDL服务，查看可视化指标。同样以DeepLab V3+模型为例：

```shell
# 指定端口号为8001
visualdl --logdir output/deeplabv3p/vdl_log --port 8001
```

服务启动后，使用浏览器打开 https://0.0.0.0:8001 或 https://localhost:8001 即可进入可视化页面。

## 模型精度验证

模型训练完成后，需要对模型进行精度验证，以确保模型的预测效果符合预期。以DeepLab V3+图像分割模型为例，可以使用以下命令启动：

```shell
# 指定需要使用的GPU设备编号
export CUDA_VISIBLE_DEVICES=0
python tutorials/eval/semantic_segmentation/deeplabv3p.py --model_path /path/to/model --save_dir /path/to/result
```

其中，`/path/to/model`是训练好的模型的保存路径，`/path/to/result`是验证结果保存路径。

## 模型部署

### 1. 导出模型

在服务端部署模型时，需要将训练过程中保存的模型导出为专用的格式。

```shell
python deploy/export/export_model.py --model_dir=./output/deeplabv3p/best_model/ --save_dir=./inference_model/
```

其中，`--model_dir`选项和`--save_dir`选项分别指定存储训练格式模型和部署格式模型的目录。例如，在上面的例子中，`./inference_model/`目录下将生成`model.pdmodel`、`model.pdiparams`、`model.pdiparams.info`、`model.yml`和`pipeline.yml`五个文件。

- `model.pdmodel`，记录模型的网络结构；
- `model.pdiparams`，包含模型权重参数；
- `model.pdiparams.info`，包含模型权重名称；
- `model.yml`，模型的配置文件（包括预处理参数、模型规格参数等）；
- `pipeline.yml`，流程配置文件。

`deploy/export/export_model.py`包含三个命令行选项：

| 参数 | 说明 |
| ---- | ---- |
| `--model_dir` | 待导出的训练格式模型存储路径，例如`./output/deeplabv3p/best_model/`。 |
| `--save_dir` | 导出的部署格式模型存储路径，例如`./inference_model/`。 |
| `--fixed_input_shape` | 固定导出模型的输入张量形状。默认值为None，表示使用任务默认输入张量形状。 |

### 2. python部署（以BIT为例）

使用预测接口的基本流程为：首先构建`Predictor`对象，然后调用`Predictor`的`predict()`方法执行预测。需要说明的是，`Predictor`对象的predict()方法返回的结果与对应的训练器（在`paddlers/tasks/`目录的文件中定义）的`predict()`方法返回结果具有相同的格式。

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
