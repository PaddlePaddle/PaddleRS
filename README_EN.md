[简体中文](README_CN.md) | English

<div align="center">
  <p align="center">
    <img src="./docs/images/logo.png" align="middle" width = "500" />
  </p>

  **A High-Performance Multi-Task Remote Sensing Toolkit Based on PaddlePaddle, Designed for End-to-End Development of Deep Learning Applications in Remote Sensing**

  [![version](https://img.shields.io/github/release/PaddlePaddle/PaddleRS.svg)](https://github.com/PaddlePaddle/PaddleRS/releases)
  [![license](https://img.shields.io/badge/license-Apache%202-blue.svg)](LICENSE)
  ![python version](https://img.shields.io/badge/python-3.7+-orange.svg)
  ![support os](https://img.shields.io/badge/os-linux%2C%20win%2C%20mac-yellow.svg)
</div>

## <img src="docs/images/seg_news_icon.png" width="30"/> News

*  [2022-11-09] 🔥 We released PaddleRS v1.0. Please check the [Release Note](https://github.com/PaddlePaddle/PaddleRS/releases).
*  [2022-05-19] 🔥 We released PaddleRS v1.0-beta. Please check the [Release Note](https://github.com/PaddlePaddle/PaddleRS/releases).

## <img src="docs/images/intro.png" width="30"/> Introduction

PaddleRS is an end-to-end high-efficent development toolkit for remote sensing applications based on PaddlePaddle, which helps both developers and researchers in the whole process of designing deep learning models, training models, optimizing performance and inference speed, and deploying models. PaddleRS supports multiple tasks, including **image segmentation, object detection, scene classification, and image restoration**.

<div align="center">
<img src="https://user-images.githubusercontent.com/71769312/218403605-fa5a9a5b-ea2a-4b82-99a0-1534e4d44328.gif"  width = "2000" />  
</div>

## <img src="./docs/images/feature.png" width="30"/> Features

* <img src="./docs/images/f1.png" width="20"/> **High-Performance Models**: PaddleRS provides 30+ deep learning models, including those reknowned in the computer vision field (e.g. DeepLab V3+, PP-YOLO) and those optimized for remote sensing tasks (e.g. BIT, FarSeg).

* <img src="./docs/images/f1.png" width="20"/> **Support for Remote Sensing Tasks**: PaddleRS supports remote sensing tasks (e.g. change detection) and provides comprehensive training, deployment tutorials, as well as rich application examples.

* <img src="./docs/images/f2.png" width="20"/> **Optimization for Large Image Tiles**: PaddleRS is optimized for the sliding window inference of large remote sensing images, using a *lazy-loading* strategy to improve performance. Also, the geospatial meta infomation for large tiles can be read and written.

* <img src="./docs/images/f2.png" width="20"/> **Data Preprocessing for Geospatial Data**: PaddleRS provides preprocessing functions for multi-spectral and multi-temporal data, which are common in the remote sensing field. PaddleRS also supports the extraction and knowledge integration of more than 50 remote sensing indices.

* <img src="./docs/images/f3.png" width="20"/> **High Efficiency**: PaddleRS provides multi-process asynchronous I/O, multi-card parallel training, evaluation, and other acceleration strategies, combined with the memory optimization function of the PaddlePaddle, which can greatly reduce the training overhead of deep learning models, all these allowing developers to train remote sensing deep learning models with a lower cost.

<div align="center">
<img src="docs/images/whole_picture.png"  width = "2000" />  
</div>

## <img src="./docs/images/chat.png" width="30"/> Community

* If you have any questions, suggestions, or feature requests, please do not hesitate to create an issue in [GitHub Issues](https://github.com/PaddlePaddle/PaddleRS/issues).
* Welcome to join PaddleRS WeChat group to communicate with us:
<div align="center">
<img src="https://user-images.githubusercontent.com/21275753/213844144-11ed841b-f71b-43a6-8e11-020883deee0a.jpg"  width = "150" />  
</div>

## <img src="./docs/images/model.png" width="30"/> Overview

<table align="center">
  <tbody>
    <tr align="center" valign="bottom">
      <td>
        <b>Models</b>
      </td>
      <td>
        <b>Data Transformation Operators</b>
      </td>
      <td>
        <b>Remote Sensing Data Tools</b>
      </td>
      <td>
        <b>Application Examples</b>
      </td>
    </tr>
    <tr valign="top">
      <td>
        <details><summary><b>Change Detection</b></summary>
        <ul>
          <li><a href="./tutorials/train/change_detection/bit.py">BIT</a></li>
          <li><a href="./tutorials/train/change_detection/cdnet.py">CDNet</a></li>
          <li><a href="./tutorials/train/change_detection/changeformer.py">ChangeFormer</a></li>
          <li><a href="./paddlers/rs_models/cd/changestar.py">ChangeStar</a></li>
          <li><a href="./tutorials/train/change_detection/dsamnet.py">DSAMNet</a></li>
          <li><a href="./tutorials/train/change_detection/dsifn.py">DSIFN</a></li>
          <li><a href="./tutorials/train/change_detection/fc_ef.py">FC-EF</a></li>
          <li><a href="./tutorials/train/change_detection/fc_siam_conc.py">FC-Siam-conc</a></li>
          <li><a href="./tutorials/train/change_detection/fc_siam_diff.py">FC-Siam-diff</a></li>
          <li><a href="./tutorials/train/change_detection/fccdn.py">FCCDN</a></li>
          <li><a href="./tutorials/train/change_detection/p2v.py">P2V-CD</a></li>
          <li><a href="./tutorials/train/change_detection/snunet.py">SNUNet</a></li>
          <li><a href="./tutorials/train/change_detection/stanet.py">STANet</a></li>
        </ul>
        </details>
        <details><summary><b>Scene Classification</b></summary>
        <ul>
          <li><a href="./tutorials/train/classification/condensenetv2.py">CondenseNet V2</a></li>
          <li><a href="./tutorials/train/classification/hrnet.py">HRNet</a></li>
          <li><a href="./tutorials/train/classification/mobilenetv3.py">MobileNetV3</a></li>
          <li><a href="./tutorials/train/classification/resnet50_vd.py">ResNet50-vd</a></li>
        </ul>
        </details>
        <details><summary><b>Image Restoration</b></summary>
        <ul>
          <li><a href="./tutorials/train/image_restoration/drn.py">DRN</a></li>
          <li><a href="./tutorials/train/image_restoration/esrgan.py">ESRGAN</a></li>
          <li><a href="./tutorials/train/image_restoration/lesrcnn.py">LESRCNN</a></li>
           <li><a href="./tutorials/train/image_restoration/nafnet.py">NAFNet</a></li>
          <li><a href="./tutorials/train/image_restoration/swinir.py">SwinIR</a></li>
        </ul>
        </details>
        <details><summary><b>Object Detection</b></summary>
        <ul>
          <li><a href="./tutorials/train/object_detection/faster_rcnn.py">Faster R-CNN</a></li>
          <li><a href="./tutorials/train/object_detection/fcosr.py">FCOSR</a></li>
          <li><a href="./tutorials/train/object_detection/ppyolo.py">PP-YOLO</a></li>
          <li><a href="./tutorials/train/object_detection/ppyolo_tiny.py">PP-YOLO Tiny</a></li>
          <li><a href="./tutorials/train/object_detection/ppyolov2.py">PP-YOLOv2</a></li>
          <li><a href="./tutorials/train/object_detection/yolov3.py">YOLOv3</a></li>
        </ul>
        </details>
        <details><summary><b>Image Segmentation</b></summary>
        <ul>
          <li><a href="./tutorials/train/semantic_segmentation/bisenetv2.py">BiSeNet V2</a></li>
          <li><a href="./tutorials/train/semantic_segmentation/deeplabv3p.py">DeepLab V3+</a></li>
          <li><a href="./tutorials/train/semantic_segmentation/factseg.py">FactSeg</a></li>
          <li><a href="./tutorials/train/semantic_segmentation/farseg.py">FarSeg</a></li>
          <li><a href="./tutorials/train/semantic_segmentation/fast_scnn.py">Fast-SCNN</a></li>
          <li><a href="./tutorials/train/semantic_segmentation/hrnet.py">HRNet</a></li>
          <li><a href="./tutorials/train/semantic_segmentation/unet.py">UNet</a></li>
        </ul>
        </details>
      </td>
      <td>
        <details><summary><b>Data Preprocessing</b></summary>
        <ul>
          <li>CenterCrop</li>
          <li>Dehaze</li>
          <li>MatchRadiance</li>
          <li>Normalize</li>
          <li>Pad</li>
          <li>ReduceDim</li>
          <li>Resize</li>
          <li>ResizeByLong</li>
          <li>ResizeByShort</li>
          <li>SelectBand</li>
          <li><a href="./docs/intro/transforms_en.md">...</a></li>
        </ul>
        </details>
        <details><summary><b>Data Augmentation</b></summary>
        <ul>
          <li>AppendIndex</li>
          <li>MixupImage</li>
          <li>RandomBlur</li>
          <li>RandomCrop</li>
          <li>RandomDistort</li>
          <li>RandomExpand</li>
          <li>RandomHorizontalFlip</li>
          <li>RandomResize</li>
          <li>RandomResizeByShort</li>
          <li>RandomScaleAspect</li>
          <li>RandomSwap</li>
          <li>RandomVerticalFlip</li>
          <li><a href="./docs/intro/transforms_en.md">...</a></li>
        </ul>
        </details>
        <details><summary><b>Remote Sensing Indices</b></summary>
        <ul>
          <li>ARI</li>
          <li>ARI2</li>
          <li>ARVI</li>
          <li>AWEInsh</li>
          <li>AWEIsh</li>
          <li>BAI</li>
          <li>BI</li>
          <li>BLFEI</li>
          <li>BNDVI</li>
          <li>BWDRVI</li>
          <li>BaI</li>
          <li>CIG</li>
          <li>CSI</li>
          <li>CSIT</li>
          <li>DBI</li>
          <li>DBSI</li>
          <li>DVI</li>
          <li>EBBI</li>
          <li>EVI</li>
          <li>EVI2</li>
          <li>FCVI</li>
          <li>GARI</li>
          <li>GBNDVI</li>
          <li>GLI</li>
          <li>GRVI</li>
          <li>IPVI</li>
          <li>LSWI</li>
          <li>MBI</li>
          <li>MGRVI</li>
          <li>MNDVI</li>
          <li>MNDWI</li>
          <li>MSI</li>
          <li>NBLI</li>
          <li>NDVI</li>
          <li>NDWI</li>
          <li>NDYI</li>
          <li>NIRv</li>
          <li>PSRI</li>
          <li>RI</li>
          <li>SAVI</li>
          <li>SWI</li>
          <li>TDVI</li>
          <li>UI</li>
          <li>VIG</li>
          <li>WI1</li>
          <li>WI2</li>
          <li>WRI</li>
          <li><a href="./docs/intro/indices_en.md">...</a></li>
        </ul>
        </details>
      </td>
      <td>
        <details><summary><b>Data Format Conversion</b></summary>
        <ul>
          <li><a href="./tools/coco2mask.py">COCO to mask</a></li>
          <li><a href="./tools/geojson2mask.py">GeoJSON to mask</a></li>
          <li><a href="./tools/mask2shape.py">mask to shapefile</a></li>
        </ul>
        </details>
        <details><summary><b>Dataset Creation Tool</b></summary>
        <ul>
          <li><a href="./tools/extract_ms_patches.py">image slicing using quadtree index</a></li>
          <li><a href="./tools/match.py">image registration</a></li>
          <li><a href="./tools/oif.py">band selection</a></li>
          <li><a href="./tools/pca.py">band fusion</a></li>
          <li><a href="./tools/split.py">image slicing</a></li>
        </ul>
        </details>
        </details>
        <details><summary><b>Data Postprocessing</b></summary>
        <ul>
          <li><a href="./paddlers/utils/postprocs/connection.py">connecting road disconnection</a></li>
          <li><a href="./paddlers/utils/postprocs/regularization.py">regularizing building boundaries</a></li>
          <li><a href="./paddlers/utils/postprocs/change_filter.py">filtering out pseudo changes</a></li>
          <li><a href="./paddlers/utils/postprocs/connection.py">connecting road disconnection</a></li>
          <li><a href="./paddlers/utils/postprocs/crf.py">refining segmentation result using a conditional random field</a></li>
          <li><a href="./paddlers/utils/postprocs/mrf.py">refining segmentation result using a Markov random field</a></li>
          <li><a href="./paddlers/utils/postprocs/regularization.py">regularizing building boundaries</a></li>
        </ul>
        </details>
        <details><summary><b>Data Visualization</b></summary>
        <ul>
          <li><a href="./paddlers/utils/visualize.py">map-raster visualization</a></li>
        </ul>
        </details>
        <details><summary><b>Preprocessing of Public Datasets</b></summary>
        <ul>
          <li><a href="./tools/prepare_dataset/prepare_levircd.py">LEVIR-CD</a></li>
          <li><a href="./tools/prepare_dataset/prepare_svcd.py">Season-varying</a></li>
          <li><a href="./tools/prepare_dataset/prepare_ucmerced.py">UC Merced</a></li>
          <li><a href="./tools/prepare_dataset/prepare_rsod.py">RSOD</a></li>
          <li><a href="./tools/prepare_dataset/prepare_isaid.py">iSAID</a></li>
          <li><a href="./docs/intro/data_prep_en.md">...</a></li>
        </ul>
      </td>
      <td>
      <details><summary><b>Official Examples</b></summary>
      <ul>
        <li><a href="./examples/rs_research/README.md">change detection research</a></li>
        <li><a href="./examples/c2fnet/README.md">small object optimization</a></li>
        <li><a href="./examples/building_extraction/README.md">building extraction</a></li>
      </ul>
      </details>
      <details><summary><b>Community Examples</b></summary>
      <ul>
      <li><a href="./examples/README.md">application examples of PaddleRS</a></li>
      </ul>
      </details>
      </td>  
    </tr>
  </tbody>
</table>

## <img src="./docs/images/teach.png" width="30"/> Tutorials and Documents

* Quick Start
  * [Quick start](./docs/quick_start_en.md)
* Data Preparation
  * [Open-source remote sensing datasets](./docs/data/dataset_en.md)
  * [Efficient interactive segmentation tool EISeg](https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.7/EISeg)
  * [Remote sensing data tools](./docs/data/tools_en.md)
* Introduction on Components
  * [Data preprocessing tools](./docs/intro/data_prep_en.md)
  * [Models](./docs/intro/model_zoo_en.md)
  * [Remote sensing indices](./docs/intro/indices_en.md)
  * [Data transforming operators](./docs/intro/transforms_en.md)
* [Model Training](./tutorials/train/README_EN.md)
* Model Deployment
  * [Model export](./deploy/export/README.md)
  * [Paddle Inference (Python)](./deploy/README.md)
  * [Interactive intelligent interprementation tool GeoView](https://github.com/PaddleCV-SIG/GeoView)
* Development and Contribution
  * [Contributing guides](./docs/CONTRIBUTING_EN.md)
  * [Development manual](./docs/dev/dev_guide_en.md)
  * [Code style guides](./docs/dev/docstring_en.md)
  * [Training APIs](./docs/apis/train_en.md)
  * [Inference APIs](./docs/apis/infer_en.md)

## <img src="./docs/images/anli.png" width="30"/> Application Examples

* [Scientific research based on PaddleRS: designing a deep learning change detection model](./examples/rs_research/README.md)
* [Optimization method for semantic segmentation of small objects in remote sensing images based on PaddleRS](./examples/c2fnet/README.md)

For more application examples, please see [application examples of PaddleRS](./examples/README.md).

## License

PaddleRS is released under the [Apache 2.0 license](./LICENSE).

## <img src="./docs/images/love.png" width="30"/> Acknowledgement

* We would like to thank the National Earth Observation Data Center, Aerospace Information Research Institute, Beihang University, Wuhan University, China University of Petroleum, China University of geosciences, China Siwei Surveying and Mapping Technology Co., Ltd., PIESAT, GEOVIS, and SuperMap for their contributions to PaddleRS(names not listed in order).
* We appreciate the contributions of [geoyee](https://github.com/geoyee), [kongdebug](https://github.com/kongdebug), and [huilin16](https://github.com/huilin16).

## <img src="./docs/images/yinyong.png" width="30"/> Citation

If you find our project useful in your research, please consider citing:

```latex
@misc{paddlers2022,
    title={PaddleRS, Awesome Remote Sensing Toolkit based on PaddlePaddle},
    author={PaddlePaddle Authors},
    howpublished = {\url{https://github.com/PaddlePaddle/PaddleRS}},
    year={2022}
}
```
