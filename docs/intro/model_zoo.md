# Model base

The base model library for PaddleRS comes from the Paddle-CV family of suites：[PaddleClas](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.3/docs/zh_CN/algorithm_introduction/ImageNet_models.md)、[PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.4/docs/model_zoo_overview_cn.md)、[PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.3/README_cn.md# model library) and [PaddleGAN] (https://github.com/Padd LePaddle/PaddleGAN/blob/develop/README_cn md# model library). In addition, PaddleRS also includes a series of remote sensing feature models, which can be used for remote sensing image segmentation and change detection.

## List of models supported by PaddleRS

PaddleRS currently supports the following models (marked with \* for remote sensing specific models) :

| Tasks | Models | Multi-band support |
|--------|---------|------|
| Change detection | \*BIT | Yes |
| Change detection | \*CDNet | Yes |
| Change detection | \*ChangeFormer | Yes |
| Change detection | \*ChangeStar | No |
| Change detection | \*DSAMNet | Yes |
| Change detection | \*DSIFN | No |
| Change detection | \*FC-EF | Yes |
| Change detection | \*FC-Siam-conc | Yes |
| Change detection | \*FC-Siam-diff | Yes |
| Change detection | \*FCCDN | Yes |
| Change detection | \*P2V-CD | Yes |
| Change detection | \*SNUNet | Yes |
| Change detection | \*STANet | Yes |
| Scene classification | CondenseNet V2 | Yes |
| Scene classification | HRNet | No |
| Scene classification | MobileNetV3 | No |
| Scene classification | ResNet50-vd | No |
| Image restoration | DRN | No |
| Image restoration | ESRGAN | Yes |
| Image restoration | LESRCNN | No |
| Object detection | Faster R-CNN | No |
| Object detection | PP-YOLO | No |
| Object detection | PP-YOLO Tiny | No |
| Object detection | PP-YOLOv2 | No |
| Object detection | YOLOv3 | No |
| Image segmentation | BiSeNet V2 | Yes |
| Image segmentation | DeepLab V3+ | Yes |
| Image segmentation | \*FactSeg | Yes |
| Image segmentation | \*FarSeg | Yes |
| Image segmentation | Fast-SCNN | Yes |
| Image segmentation | HRNet | Yes |
| Image segmentation | UNet | Yes |
