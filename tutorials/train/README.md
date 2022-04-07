# 使用教程——训练模型

本目录下整理了使用PaddleRS训练模型的示例代码。代码中均提供了示例数据的自动下载，并均使用GPU对模型进行训练。

|示例代码路径 | 任务 | 模型 |
|------|--------|---------|
|change_detection/bit.py | 变化检测 | BIT |
|change_detection/cdnet.py | 变化检测 | CDNet |
|change_detection/dsamnet.py | 变化检测 | DSAMNet |
|change_detection/dsifn.py | 变化检测 | DSIFN |
|change_detection/snunet.py | 变化检测 | SNUNet |
|change_detection/stanet.py | 变化检测 | STANet |
|change_detection/fc_ef.py | 变化检测 | FC-EF |
|change_detection/fc_siam_conc.py | 变化检测 | FC-Siam-conc |
|change_detection/fc_siam_diff.py | 变化检测 | FC-Siam-diff |
|classification/hrnet.py | 场景分类 | HRNet |
|classification/mobilenetv3.py | 场景分类 | MobileNetV3 |
|classification/resnet50_vd.py | 场景分类 | ResNet50-vd |
|image_restoration/drn.py | 超分辨率 | DRN |
|image_restoration/esrgan.py | 超分辨率 | ESRGAN |
|image_restoration/lesrcnn.py | 超分辨率 | LESRCNN |
|object_detection/faster_rcnn.py | 目标检测 | Faster R-CNN |
|object_detection/ppyolo.py | 目标检测 | PP-YOLO |
|object_detection/ppyolotiny.py | 目标检测 | PP-YOLO Tiny |
|object_detection/ppyolov2.py | 目标检测 | PP-YOLOv2 |
|object_detection/yolov3.py | 目标检测 | YOLOv3 |
|semantic_segmentation/deeplabv3p.py | 图像分割 | DeepLab V3+ |
|semantic_segmentation/unet.py | 图像分割 | UNet |

<!-- 可参考API接口说明了解示例代码中的API：
* [数据集读取API](../../docs/apis/datasets.md)
* [数据预处理和数据增强API](../../docs/apis/transforms/transforms.md)
* [模型API/模型加载API](../../docs/apis/models/README.md)
* [预测结果可视化API](../../docs/apis/visualize.md) -->

## 环境准备

- [PaddlePaddle安装](https://www.paddlepaddle.org.cn/install/quick)
  * 版本要求：PaddlePaddle>=2.2.0

- PaddleRS安装

PaddleRS代码会跟随开发进度不断更新，可以安装develop分支的代码使用最新的功能，安装方式如下：

```
git clone https://github.com/PaddleCV-SIG/PaddleRS
cd PaddleRS
git checkout develop
pip install -r requirements.txt
python setup.py install
```

- \*GDAL安装

PaddleRS支持多种类型的卫星数据IO以及地理处理等，可能需要使用GDAL，可以根据需求进行安装，安装方式如下：

  - Linux / MacOS

推荐使用conda进行安装:

```
conda install gdal
```

  - Windows

Windows用户可以通过[这里](https://www.lfd.uci.edu/~gohlke/pythonlibs/#gdal)下载对应Python和系统版本的二进制文件（\*.whl）到本地，以*GDAL‑3.3.3‑cp39‑cp39‑win_amd64.whl*为例，进入下载目录进行安装:

```
cd download
pip install GDAL‑3.3.3‑cp39‑cp39‑win_amd64.whl
```

## 开始训练
* 在安装PaddleRS后，使用如下命令进行单卡训练。代码会自动下载训练数据。以DeepLab V3+图像分割模型为例：

```commandline
# 指定需要使用的GPU设备编号
export CUDA_VISIBLE_DEVICES=0
python tutorials/train/semantic_segmentation/deeplabv3p.py
```

* 如需使用多块GPU进行训练，例如使用2张显卡时，执行如下命令：

```commandline
python -m paddle.distributed.launch --gpus 0,1 tutorials/train/semantic_segmentation/deeplabv3p.py
```

## VisualDL可视化训练指标
将传入`train`方法的`use_vdl`参数设为`True`，则模型训练过程中将自动把训练日志以VisualDL的格式存储到`save_dir`（用户自己指定的路径）目录下名为`vdl_log`的子目录中。用户可以使用如下命令启动VisualDL服务，查看可视化指标。同样以DeepLab V3+模型为例：
```commandline
# 指定端口号为8001
visualdl --logdir output/deeplabv3p/vdl_log --port 8001
```

服务启动后，使用浏览器打开 https://0.0.0.0:8001 或 https://localhost:8001
