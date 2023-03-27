# 快速开始

本目录中整理了使用PaddleRS训练模型的示例代码。代码中均提供对示例数据的自动下载，并均使用GPU对模型进行训练。

| 示例代码路径                              | 任务 | 模型 |
|-------------------------------------|--------|---------|
| ./tutorials/change_detection/bit.py | 变化检测 | BIT |
| ./tutorials/change_detection/cdnet.py           | 变化检测 | CDNet |
| ./tutorials/change_detection/changeformer.py    | 变化检测 | ChangeFormer |
| ./tutorials/change_detection/dsamnet.py         | 变化检测 | DSAMNet |
| ./tutorials/change_detection/dsifn.py           | 变化检测 | DSIFN |
| ./tutorials/change_detection/fc_ef.py           | 变化检测 | FC-EF |
| ./tutorials/change_detection/fc_siam_conc.py    | 变化检测 | FC-Siam-conc |
| ./tutorials/change_detection/fc_siam_diff.py    | 变化检测 | FC-Siam-diff |
| ./tutorials/change_detection/fccdn.py           | 变化检测 | FCCDN |
| ./tutorials/change_detection/p2v.py             | 变化检测 | P2V-CD |
| ./tutorials/change_detection/snunet.py          | 变化检测 | SNUNet |
| ./tutorials/change_detection/stanet.py          | 变化检测 | STANet |
| ./tutorials/classification/condensenetv2.py     | 场景分类 | CondenseNet V2 |
| ./tutorials/classification/hrnet.py             | 场景分类 | HRNet |
| ./tutorials/classification/mobilenetv3.py       | 场景分类 | MobileNetV3 |
| ./tutorials/classification/resnet50_vd.py       | 场景分类 | ResNet50-vd |
| ./tutorials/image_restoration/drn.py            | 图像复原 | DRN |
| ./tutorials/image_restoration/esrgan.py         | 图像复原 | ESRGAN |
| ./tutorials/image_restoration/lesrcnn.py        | 图像复原 | LESRCNN |
| ./tutorials/object_detection/faster_rcnn.py     | 目标检测 | Faster R-CNN |
| ./tutorials/object_detection/ppyolo.py          | 目标检测 | PP-YOLO |
| ./tutorials/object_detection/ppyolo_tiny.py     | 目标检测 | PP-YOLO Tiny |
| ./tutorials/object_detection/ppyolov2.py        | 目标检测 | PP-YOLOv2 |
| ./tutorials/object_detection/yolov3.py          | 目标检测 | YOLOv3 |
| ./tutorials/semantic_segmentation/bisenetv2.py  | 图像分割 | BiSeNet V2 |
| ./tutorials/semantic_segmentation/deeplabv3p.py | 图像分割 | DeepLab V3+ |
| ./tutorials/semantic_segmentation/factseg.py    | 图像分割 | FactSeg |
| ./tutorials/semantic_segmentation/farseg.py     | 图像分割 | FarSeg |
| ./tutorials/semantic_segmentation/fast_scnn.py  | 图像分割 | Fast-SCNN |
| ./tutorials/semantic_segmentation/hrnet.py      | 图像分割 | HRNet |
| ./tutorials/semantic_segmentation/unet.py       | 图像分割 | UNet |

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

## 模型训练

+ 在安装完成PaddleRS后，即可开始模型训练。
+ 模型训练可参考：[使用教程——训练模型](./tutorials/train/README.md)

## 模型精度验证

模型训练完成后，需要对模型进行精度验证，以确保模型的预测效果符合预期。以DeepLab V3+图像分割模型为例，可以使用以下命令启动：

```python
import paddlex as pdx

# 加载模型
model = pdx.load_model('output/deeplabv3p/best_model')

# 加载验证集
dataset = pdx.datasets.SegDataset(
    data_dir='dataset/val',
    file_list='dataset/val/list.txt',
    label_list='dataset/labels.txt',
    transforms=model.eval_transforms)

# 进行验证
result = model.evaluate(dataset, batch_size=1, epoch_id=None, return_details=True)

print(result)
```

在上述代码中，`pdx.load_model()`方法用于加载预训练的DeepLabV3P模型，`pdx.datasets.SegDataset()`方法用于加载验证集数据。`model.evaluate()`方法接受验证集数据集、批大小和轮数等参数，并返回包括预测结果和指标评估在内的验证结果。最后，我们可以打印输出验证结果。


## 模型部署

### 1. 导出模型

导出模型可见：[模型导出](./deploy/export/README.md)

### 2. python部署

python部署可见：[Python部署](./deploy/README.md)
