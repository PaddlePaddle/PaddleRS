# Tutorial - Training model

Sample code using the PaddleRS training model is curated in this directory. The code provides automatic downloading of sample data, and uses GPU to train the model.

|Sample code path | Task | Model |
|------|--------|---------|
|change_detection/bit.py | Change Detection | BIT |
|change_detection/cdnet.py | Change Detection | CDNet |
|change_detection/changeformer.py | Change Detection | ChangeFormer |
|change_detection/dsamnet.py | Change Detection | DSAMNet |
|change_detection/dsifn.py | Change Detection | DSIFN |
|change_detection/fc_ef.py | Change Detection | FC-EF |
|change_detection/fc_siam_conc.py | Change Detection | FC-Siam-conc |
|change_detection/fc_siam_diff.py | Change Detection | FC-Siam-diff |
|change_detection/fccdn.py | Change Detection | FCCDN |
|change_detection/p2v.py | Change Detection | P2V-CD |
|change_detection/snunet.py | Change Detection | SNUNet |
|change_detection/stanet.py | Change Detection | STANet |
|classification/condensenetv2.py | Scene Classification | CondenseNet V2 |
|classification/hrnet.py | Scene Classification | HRNet |
|classification/mobilenetv3.py | Scene Classification | MobileNetV3 |
|classification/resnet50_vd.py | Scene Classification | ResNet50-vd |
|image_restoration/drn.py | Image Restoration | DRN |
|image_restoration/esrgan.py | Image Restoration | ESRGAN |
|image_restoration/lesrcnn.py | Image Restoration | LESRCNN |
|object_detection/faster_rcnn.py | Object Detection | Faster R-CNN |
|object_detection/ppyolo.py | Object Detection | PP-YOLO |
|object_detection/ppyolo_tiny.py | Object Detection | PP-YOLO Tiny |
|object_detection/ppyolov2.py | Object Detection | PP-YOLOv2 |
|object_detection/yolov3.py | Object Detection | YOLOv3 |
|semantic_segmentation/bisenetv2.py | Image Segmentation | BiSeNet V2 |
|semantic_segmentation/deeplabv3p.py | Image Segmentation | DeepLab V3+ |
|semantic_segmentation/factseg.py | Image Segmentation | FactSeg |
|semantic_segmentation/farseg.py | Image Segmentation | FarSeg |
|semantic_segmentation/fast_scnn.py | Image Segmentation | Fast-SCNN |
|semantic_segmentation/hrnet.py | Image Segmentation | HRNet |
|semantic_segmentation/unet.py | Image Segmentation | UNet |

## Environmental preparation

+ [PaddlePaddle installation](https://www.paddlepaddle.org.cn/install/quick)
  - Version requirements: PaddlePaddle>=2.2.0

+ PaddleRS installation

The PaddleRS code will be updated as the development progresses. You can install code for the develop branch to use the latest features as follows:

```shell
git clone https://github.com/PaddlePaddle/PaddleRS
cd PaddleRS
git checkout develop
pip install -r requirements.txt
python setup.py install
```

If the download dependence is slow or times out when using `python setup.py install`, you can create `setup.cfg` in the same directory as `setup.py` and input the following content, then the download can be accelerated through Tsinghua source:

```
[easy_install]
index-url=https://pypi.tuna.tsinghua.edu.cn/simple
```

+ (Optional) GDAL installation

PaddleRS supports reading of various types of satellite data. To fully use PaddleRS remote sensing data reading function, you need to install GDAL as follows:

  - Linux / MacOS

conda is recommended for installation:

```shell
conda install gdal
```

  - Windows

Windows users can download the Python and system version corresponding to the .whl format installation package from [this](https://www.lfd.uci.edu/~gohlke/pythonlibs/#gdal) to local, take *GDAL‑3.3.3‑cp39‑cp39‑win_amd64.whl* as an example, use the pip tool installation:

```shell
pip install GDAL‑3.3.3‑cp39‑cp39‑win_amd64.whl
```

### *Docker installation

1. Pull from dockerhub:

```shell
docker pull paddlepaddle/paddlers:1.0.0
```

- (Optional) Build from scratch. Multiple base images for PaddlePaddle can be selected by setting `PPTAG` to build cpus or different GPU environments:

```shell
git clone https://github.com/PaddlePaddle/PaddleRS
cd PaddleRS
docker build -t <imageName> .  # default is 2.4.1-cpu version
# docker build -t <imageName> . --build-arg PPTAG=2.4.1-gpu-cuda10.2-cudnn7.6-trt7.0  # One of the gpu versions of 2.4.1
# Other Tag refer to: https://hub.docker.com/r/paddlepaddle/paddle/tags
```

2. Start image

```shell
docker iamges  # View the ID of an image
docker run -it <imageID>
```

## Start training

+ After PaddleRS is installed, run the following command to perform single-card training. The script will automatically download the training data. Take DeepLab V3+ image segmentation model as an example:

```shell
# Specifies the GPU device number to be used
export CUDA_VISIBLE_DEVICES=0
python tutorials/train/semantic_segmentation/deeplabv3p.py
```

+ If multiple Gpus are required for training, for example, two graphics cards, run the following command:

```shell
python -m paddle.distributed.launch --gpus 0,1 tutorials/train/semantic_segmentation/deeplabv3p.py
```

## VisualDL Visual training metrics

Set the `use_vdl` parameter passed to the `train()` method to `True`, then the training log will be automatically stored in the format of `save_dir` (user specified path) in the subdirectory named `vdl_log` during the model training process. You can run the following command to start the VisualDL service and view visual indicators. DeepLab V3+ model is also taken as an example:

```shell
# The specified port number is 8001
visualdl --logdir output/deeplabv3p/vdl_log --port 8001
```

Once the service is started, use your browser to open https://0.0.0.0:8001 or https://localhost:8001 to access the visual page.
