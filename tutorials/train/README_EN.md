[简体中文](README_CN.md) | English

# Tutorials - Model Training

Sample code using the PaddleRS training model is curated in this directory. The code provides automatic downloading of sample data, and uses GPU to train the model.

|Sample Code Path | Task | Model |
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
|object_detection/fcosr.py | Object Detection | FCOSR |
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

## Start Training

+ After PaddleRS is installed, run the following commands to launch training with a single GPU. The script will automatically download the training data. Take DeepLab V3+ image segmentation model as an example:

```shell
# Specifies the GPU device number to be used
export CUDA_VISIBLE_DEVICES=0
python tutorials/train/semantic_segmentation/deeplabv3p.py
```

+ If multiple GPUs are required for training, for example, two graphics cards, run the following command:

```shell
python -m paddle.distributed.launch --gpus 0,1 tutorials/train/semantic_segmentation/deeplabv3p.py
```

## Visualize Training Metrics via VisualDL

Set the `use_vdl` argument passed to the `train()` method to `True`, and then the training log will be automatically saved in VisualDL format in a subdirectory named `vdl_log` under the directory specified by `save_dir` (a user-specified path) during the model training process. You can run the following command to start the VisualDL service and view the indicators and metrics. We also take DeepLab V3+ as an example:

```shell
# The specified port number is 8001
visualdl --logdir output/deeplabv3p/vdl_log --port 8001
```

Once the service is started, open https://0.0.0.0:8001 or https://localhost:8001 in your browser to access the VisualDL page.
