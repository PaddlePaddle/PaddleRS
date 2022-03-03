# 使用教程——训练模型

本目录下整理了使用PaddleRS训练模型的示例代码，代码中均提供了示例数据的自动下载，并均使用单张GPU卡进行训练。

|代码 | 模型任务 | 数据 |
|------|--------|---------|
|object_detection/ppyolo.py | 目标检测PPYOLO | 昆虫检测 |
|semantic_segmentation/deeplabv3p_resnet50_vd.py | 语义分割DeepLabV3 | 视盘分割 |

<!-- 可参考API接口说明了解示例代码中的API：
* [数据集读取API](../../docs/apis/datasets.md)
* [数据预处理和数据增强API](../../docs/apis/transforms/transforms.md)
* [模型API/模型加载API](../../docs/apis/models/README.md)
* [预测结果可视化API](../../docs/apis/visualize.md) -->


# 环境准备

- [PaddlePaddle安装](https://www.paddlepaddle.org.cn/install/quick)
* 版本要求：PaddlePaddle>=2.1.0

<!-- - [PaddleRS安装](../../docs/install.md) -->

## 开始训练
* 修改tutorials/train/semantic_segmentation/deeplabv3p_resnet50_vd.py中sys.path路径
```
sys.path.append("your/PaddleRS/path")
```

* 在安装PaddleRS后，使用如下命令开始训练，代码会自动下载训练数据, 并均使用单张GPU卡进行训练。

```commandline
export CUDA_VISIBLE_DEVICES=0
python tutorials/train/semantic_segmentation/deeplabv3p_resnet50_vd.py
```

* 若需使用多张GPU卡进行训练，例如使用2张卡时执行：

```commandline
python -m paddle.distributed.launch --gpus 0,1 tutorials/train/semantic_segmentation/deeplabv3p_resnet50_vd.py
```
使用多卡时，参考[训练参数调整](../../docs/parameters.md)调整学习率和批量大小。


## VisualDL可视化训练指标
在模型训练过程，在`train`函数中，将`use_vdl`设为True，则训练过程会自动将训练日志以VisualDL的格式打点在`save_dir`（用户自己指定的路径）下的`vdl_log`目录，用户可以使用如下命令启动VisualDL服务，查看可视化指标
```commandline
visualdl --logdir output/deeplabv3p_resnet50_vd/vdl_log --port 8001
```

服务启动后，使用浏览器打开 https://0.0.0.0:8001 或 https://localhost:8001


