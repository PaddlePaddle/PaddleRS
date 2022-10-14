# 飞桨训推一体全流程（TIPC）

## 1 简介

飞桨除了基本的模型训练和预测，还提供了支持多端多平台的高性能推理部署工具。本文档提供了飞桨训推一体全流程（Training and Inference Pipeline Criterion(TIPC)）信息和测试工具，方便用户查阅每种模型的训练推理部署打通情况，并可以进行一键测试。

<div align="center">
    <img src="docs/overview.png" width="1000">
</div>

## 2 汇总信息

打通情况汇总如下，已填写的部分表示可以使用本工具进行一键测试，未填写的表示正在支持中。

**字段说明：**
- 基础训练预测：指Linux GPU/CPU环境下的模型训练、Paddle Inference Python预测。
- 更多训练方式：包括多机多卡、混合精度训练。
- 更多部署方式：包括C++预测、Serving服务化部署、ARM端侧部署等多种部署方式，具体列表见[3.3节](#3.3)
- Slim训练部署：包括PACT在线量化、离线量化。
- 更多训练环境：包括Windows GPU/CPU、Linux NPU、Linux DCU等多种环境。


| 任务类别 | 模型名称 | 基础<br>训练预测 | 更多<br>训练方式 | 更多<br>部署方式 | Slim<br>训练部署 |  更多<br>训练环境  |
| :--- | :--- |  :----:  | :--------: |  :----:  |   :----:  |   :----:  |
| 变化检测 | BIT | 支持 | - | - | - |
| 变化检测 | CDNet | 支持 | - | - | - |
| 变化检测 | DSAMNet | 支持 | - | - | - |
| 变化检测 | DSIFN | 支持 | - | - | - |
| 变化检测 | SNUNet | 支持 | - | - | - |
| 变化检测 | STANet | 支持 | - | - | - |
| 变化检测 | FC-EF | 支持 | - | - | - |
| 变化检测 | FC-Siam-conc | 支持 | - | - | - |
| 变化检测 | FC-Siam-diff | 支持 | - | - | - |
| 变化检测 | ChangeFormer | 支持 | - | - | - |
| 场景分类 | CondenseNet V2 | 支持 | - | - | - |
| 场景分类 | HRNet | 支持 | - | - | - |
| 场景分类 | MobileNetV3 | 支持 | - | - | - |
| 场景分类 | ResNet50-vd | 支持 | - | - | - |
| 图像复原 | DRN | 支持 | - | - | - |
| 图像复原 | EARGAN | 支持 | - | - | - |
| 图像复原 | LESRCNN | 支持 | - | - | - |
| 目标检测 | Faster R-CNN | 支持 | - | - | - |
| 目标检测 | PP-YOLO | 支持 | - | - | - |
| 目标检测 | PP-YOLO Tiny | 支持 | - | - | - |
| 目标检测 | PP-YOLOv2 | 支持 | - | - | - |
| 目标检测 | YOLOv3 | 支持 | - | - | - |
| 图像分割 | BiSeNet V2 | 支持 | - | - | - |
| 图像分割 | DeepLab V3+ | 支持 | - | - | - |
| 图像分割 | FactSeg | 支持 | - | - | - |
| 图像分割 | FarSeg | 支持 | - | - | - |
| 图像分割 | Fast-SCNN | 支持 | - | - | - |
| 图像分割 | HRNet | 支持 | - | - | - |
| 图像分割 | UNet | 支持 | - | - | - |

## 3 测试工具简介

### 3.1 目录介绍

```
test_tipc
    |--configs                                      # 配置目录
    |    |--task_name                               # 任务名称
    |           |--model_name                       # 模型名称
    |                   |--train_infer_python.txt   # 基础训练推理测试配置文件
    |--docs                                         # 文档目录
    |   |--test_train_inference_python.md           # 基础训练推理测试说明文档
    |----README.md                                  # TIPC说明文档
    |----prepare.sh                                 # TIPC基础训练推理测试数据准备脚本
    |----test_train_inference_python.sh             # TIPC基础训练推理测试解析脚本
    |----common_func.sh                             # TIPC基础训练推理测试常用函数
```

### 3.2 测试流程概述

使用本工具，可以测试不同功能的支持情况。测试过程包含：

1. 准备数据与环境；
2. 运行测试脚本，观察不同配置是否运行成功。

<a name="3.3"></a>
### 3.3 开始测试

请参考相应文档，完成指定功能的测试。

- 基础训练预测测试：
    - [Linux GPU/CPU 基础训练推理测试](docs/test_train_inference_python.md)
