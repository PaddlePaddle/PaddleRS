# Linux GPU/CPU 基础训练推理测试

Linux GPU/CPU 基础训练推理测试的主程序为`test_train_inference_python.sh`，可以测试基于Python的模型训练、评估、推理等基本功能。

## 1 测试结论汇总

- 训练相关：

| 任务类别 | 模型名称 | 单机单卡 | 单机多卡 | 参考预测精度 |
| :----: | :----: | :----: | :----: | :----: |
|  变化检测  | BIT | 正常训练 | 正常训练 | IoU=71.01% |
|  变化检测  | CDNet | 正常训练 | 正常训练 | IoU=55.10% |
|  变化检测  | ChangeFormer | 正常训练 | 正常训练 | IoU=61.09% |
|  变化检测  | DSAMNet | 正常训练 | 正常训练 | IoU=69.02% |
|  变化检测  | DSIFN | 正常训练 | 正常训练 | IoU=72.36% |
|  变化检测  | FC-EF | 正常训练 | 正常训练 | IoU=57.18% |
|  变化检测  | FC-Siam-conc | 正常训练 | 正常训练 | IoU=52.82% |
|  变化检测  | FC-Siam-diff | 正常训练 | 正常训练 | IoU=58.30% |
|  变化检测  | FCCDN | 正常训练 | 正常训练 | IoU=23.94% |
|  变化检测  | SNUNet | 正常训练 | 正常训练 | IoU=67.66% |
|  变化检测  | STANet | 正常训练 | 正常训练 | IoU=67.23% |
|  场景分类  | CondenseNet V2 | 正常训练 | 正常训练 | Acc(top1)=60.53% |
|  场景分类  | HRNet | 正常训练 | 正常训练 | Acc(top1)=99.47% |
|  场景分类  | MobileNetV3 | 正常训练 | 正常训练 | Acc(top1)=99.57% |
|  场景分类  | ResNet50-vd | 正常训练 | 正常训练 | Acc(top1)=99.37% |
|  目标检测  | Faster R-CNN | 正常训练 | 正常训练 | 暂无稳定精度 |
|  目标检测  | PP-YOLO | 正常训练 | 正常训练 | 暂无稳定精度 |
|  目标检测  | PP-YOLO Tiny | 正常训练 | 正常训练 | 暂无稳定精度 |
|  目标检测  | PP-YOLOv2 | 正常训练 | 正常训练 | 暂无稳定精度 |
|  目标检测  | YOLOv3 | 正常训练 | 正常训练 | 暂无稳定精度 |
|  图像复原  | DRN | 正常训练 | 正常训练 | PSNR=24.92 |
|  图像复原  | ESRGAN | 正常训练 | 正常训练 | PSNR=21.39 |
|  图像复原  | LESRCNN | 正常训练 | 正常训练 | PSNR=23.67 |
|  图像分割  | BiSeNet V2 | 正常训练 | 正常训练 | mIoU=70.52% |
|  图像分割  | DeepLab V3+ | 正常训练 | 正常训练 | mIoU=64.41% |
|  图像分割  | FactSeg | 正常训练 | 正常训练 |  |
|  图像分割  | FarSeg | 正常训练 | 正常训练 | mIoU=50.60% |
|  图像分割  | Fast-SCNN | 正常训练 | 正常训练 | mIoU=49.27% |
|  图像分割  | HRNet | 正常训练 | 正常训练 | mIoU=33.03% |
|  图像分割  | UNet | 正常训练 | 正常训练 | mIoU=72.58% |

*注：参考预测精度为whole_train_whole_infer模式下单卡训练汇报的精度数据。*

- 推理相关：

| 任务类别 | 模型名称 | device_CPU | device_GPU | batchsize |
|  :----:   |  :----: |   :----:   |  :----:  |   :----:   |
|  变化检测  | BIT | 支持 | 支持 | 1 |
|  变化检测  | CDNet | 支持 | 支持 | 1 |
|  变化检测  | ChangeFormer | 支持 | 支持 | 1 |
|  变化检测  | DSAMNet | 支持 | 支持 | 1 |
|  变化检测  | DSIFN | 支持 | 支持 | 1 |
|  变化检测  | SNUNet | 支持 | 支持 | 1 |
|  变化检测  | STANet | 支持 | 支持 | 1 |
|  变化检测  | FC-EF | 支持 | 支持 | 1 |
|  变化检测  | FC-Siam-conc | 支持 | 支持 | 1 |
|  变化检测  | FC-Siam-diff | 支持 | 支持 | 1 |
|  场景分类  | CondenseNet V2 | 支持 | 支持 | 1 |
|  场景分类  | HRNet | 支持 | 支持 | 1 |
|  场景分类  | MobileNetV3 | 支持 | 支持 | 1 |
|  场景分类  | ResNet50-vd | 支持 | 支持 | 1 |
|  图像复原  | DRN | 支持 | 支持 | 1 |
|  图像复原  | ESRGAN | 支持 | 支持 | 1 |
|  图像复原  | LESRCNN | 支持 | 支持 | 1 |
|  目标检测  | Faster R-CNN | 支持 | 支持 | 1 |
|  目标检测  | PP-YOLO | 支持 | 支持 | 1 |
|  目标检测  | PP-YOLO Tiny | 支持 | 支持 | 1 |
|  目标检测  | PP-YOLOv2 | 支持 | 支持 | 1 |
|  目标检测  | YOLOv3 | 支持 | 支持 | 1 |
|  图像分割  | BiSeNet V2 | 支持 | 支持 | 1 |
|  图像分割  | DeepLab V3+ | 支持 | 支持 | 1 |
|  图像分割  | FactSeg | 支持 | 支持 | 1 |
|  图像分割  | FarSeg | 支持 | 支持 | 1 |
|  图像分割  | Fast-SCNN | 支持 | 支持 | 1 |
|  图像分割  | HRNet | 支持 | 支持 | 1 |
|  图像分割  | UNet | 支持 | 支持 | 1 |

## 2 测试流程

### 2.1 环境配置

除了安装PaddleRS以外，您还需要安装规范化日志输出工具AutoLog：
```
pip install  https://paddleocr.bj.bcebos.com/libs/auto_log-1.2.0-py3-none-any.whl
```

### 2.2 功能测试

先运行`test_tipc/prepare.sh`准备数据和模型，然后运行`test_tipc/test_train_inference_python.sh`进行测试。测试过程中生成的日志文件均存储在`test_tipc/output/`目录。

`test_tipc/test_train_inference_python.sh`支持4种运行模式，分别是：

- 模式1：lite_train_lite_infer，使用少量数据训练，用于快速验证训练到预测的流程是否能走通，不验证精度和速度；
```shell
bash ./test_tipc/prepare.sh test_tipc/configs/clas/hrnet/train_infer_python.txt lite_train_lite_infer
bash ./test_tipc/test_train_inference_python.sh test_tipc/configs/clas/hrnet/train_infer_python.txt lite_train_lite_infer
```  

- 模式2：lite_train_whole_infer，使用少量数据训练，全量数据预测，用于验证训练后的模型执行预测时预测速度是否合理；
```shell
bash ./test_tipc/prepare.sh test_tipc/configs/clas/hrnet/train_infer_python.txt lite_train_whole_infer
bash ./test_tipc/test_train_inference_python.sh test_tipc/configs/clas/hrnet/train_infer_python.txt lite_train_whole_infer
```  

- 模式3：whole_infer，不训练，使用全量数据预测，验证模型动转静是否正常，检查模型的预测时间和精度;
```shell
bash ./test_tipc/prepare.sh test_tipc/configs/clas/hrnet/train_infer_python.txt whole_infer
# 用法1:
bash ./test_tipc/test_train_inference_python.sh test_tipc/configs/clas/hrnet/train_infer_python.txt whole_infer
# 用法2: 在指定GPU上执行预测，第三个传入参数为GPU编号
bash ./test_tipc/test_train_inference_python.sh test_tipc/configs/clas/hrnet/train_infer_python.txt whole_infer '1'
```  

- 模式4：whole_train_whole_infer，CE： 全量数据训练，全量数据预测，验证模型训练精度、预测精度、预测速度；
```shell
bash ./test_tipc/prepare.sh test_tipc/configs/clas/hrnet/train_infer_python.txt whole_train_whole_infer
bash ./test_tipc/test_train_inference_python.sh test_tipc/configs/clas/hrnet/train_infer_python.txt whole_train_whole_infer
```  

运行相应指令后，在`test_tipc/output`目录中会自动保存运行日志。如lite_train_lite_infer模式下，该目录中可能存在以下文件：
```
test_tipc/output/{task name}/{model name}/
|- results_python.log    # 存储指令执行状态的日志
|- norm_gpus_0_autocast_null/  # GPU 0号卡上的训练日志和模型保存目录
......
|- python_infer_cpu_usemkldnn_True_threads_6_precision_fp32_batchsize_1.log  # CPU上开启mkldnn，线程数设置为6，测试batch_size=1条件下的预测运行日志
|- python_infer_gpu_usetrt_True_precision_fp16_batchsize_1.log # GPU上开启TensorRT，测试batch_size=1的半精度预测运行日志
......
```

其中`results_python.log`中保存了每条指令的执行状态。如果指令运行成功，输出信息如下所示：
```
 Run successfully with command - hrnet - python test_tipc/infer.py --file_list ./test_tipc/data/ucmerced/ ./test_tipc/data/ucmerced/val.txt --device=gpu --use_trt=False --precision=fp32 --model_dir=./test_tipc/output/clas/hrnet/lite_train_lite_infer/norm_gpus_0,1_autocast_null/static/ --batch_size=1 --benchmark=True  !
......
```

如果运行失败，输出信息如下所示：
```
 Run failed with command - hrnet - python test_tipc/infer.py --file_list ./test_tipc/data/ucmerced/ ./test_tipc/data/ucmerced/val.txt --device=gpu --use_trt=False --precision=fp32 --model_dir=./test_tipc/output/clas/hrnet/lite_train_lite_infer/norm_gpus_0,1_autocast_null/static/ --batch_size=1 --benchmark=True  !
......
```
