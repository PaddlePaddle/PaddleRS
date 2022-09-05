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
|change_detection/snunet.py | 变化检测 | SNUNet |
|change_detection/stanet.py | 变化检测 | STANet |
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
|semantic_segmentation/deeplabv3p.py | 图像分割 | DeepLab V3+ |
|semantic_segmentation/farseg.py | 图像分割 | FarSeg |
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
