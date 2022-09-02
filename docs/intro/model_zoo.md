# 模型库

PaddleRS的基础模型库来自Paddle-CV系列套件：[PaddleClas](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.3/docs/zh_CN/algorithm_introduction/ImageNet_models.md)、[PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.4/docs/model_zoo_overview_cn.md)、[PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.3/README_cn.md#模型库)以及[PaddleGAN](https://github.com/PaddlePaddle/PaddleGAN/blob/develop/README_cn.md#模型库)。除此之外，PaddleRS也包含一系列遥感特色模型，可用于遥感影像分割、变化检测等。

## PaddleRS已支持的模型列表

PaddleRS目前已支持的全部模型如下（标注\*的为遥感专用模型）：

| 任务 | 模型 | 多波段支持 |
|--------|---------|------|
| 变化检测 | \*BIT | 是 |
| 变化检测 | \*CDNet | 是 |
| 变化检测 | \*ChangeFormer | 是 |
| 变化检测 | \*ChangeStar | 否 |
| 变化检测 | \*DSAMNet | 是 |
| 变化检测 | \*DSIFN | 否 |
| 变化检测 | \*FC-EF | 是 |
| 变化检测 | \*FC-Siam-conc | 是 |
| 变化检测 | \*FC-Siam-diff | 是 |
| 变化检测 | \*FCCDN | 是 |
| 变化检测 | \*SNUNet | 是 |
| 变化检测 | \*STANet | 是 |
| 场景分类 | CondenseNetV2 | 是 |
| 场景分类 | HRNet | 是 |
| 场景分类 | MobileNetV3 | 是 |
| 场景分类 | ResNet50-vd | 是 |
| 图像复原 | DRN | 否 |
| 图像复原 | ESRGAN | 否 |
| 图像复原 | LESRCNN | 否 |
| 目标检测 | Faster R-CNN | 是 |
| 目标检测 | PP-YOLO | 是 |
| 目标检测 | PP-YOLO Tiny | 是 |
| 目标检测 | PP-YOLOv2 | 是 |
| 目标检测 | YOLOv3 | 是 |
| 图像分割 | DeepLab V3+ | 是 |
| 图像分割 | \*FarSeg | 否 |
| 图像分割 | UNet | 是 |
