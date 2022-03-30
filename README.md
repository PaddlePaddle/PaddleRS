简体中文 | [English](README_EN.md)

<div align="center">
  <p align="center">
    <img src="./docs/images/logo.png" align="middle" width = "500" />
  </p>

  **飞桨高性能遥感图像处理开发套件，端到端地完成从训练到部署的全流程遥感深度学习应用。**

  <!-- [![Build Status](https://travis-ci.org/PaddleCV-SIG/PaddleRS.svg?branch=release/0.1)](https://travis-ci.org/PaddleCV-SIG/PaddleRS) -->
  <!-- [![Version](https://img.shields.io/github/release/PaddleCV-SIG/PaddleRS.svg)](https://github.com/PaddleCV-SIG/PaddleRS/releases) -->
  [![License](https://img.shields.io/badge/license-Apache%202-blue.svg)](LICENSE)
  ![python version](https://img.shields.io/badge/python-3.6+-orange.svg)
  ![support os](https://img.shields.io/badge/os-linux%2C%20win%2C%20mac-yellow.svg)
</div>

## 最新动态 <img src="docs/images/seg_news_icon.png" width="30"/>

* [2022-03-30] PaddleRS alpha版本发布！详细发版信息请参考[Release Note](https://github.com/PaddleCV-SIG)。

## 简介

PaddleRS是xxx、xxx、xxx等遥感科研院所共同基于飞桨开发的遥感处理平台，支持遥感图像分类，目标检测，图像分割，以及变化检测等常用遥感任务，帮助开发者更便捷地完成从训练到部署全流程遥感深度学习应用。

<div align="center">
<img src="docs/images/whole_image.jpg"  width = "2000" />  
</div>

## 特性 <img src="./docs/images/feature.png" width="30"/>

* <img src="./docs/images/f1.png" width="20"/> **特有的遥感数据处理模块**：针对遥感行业数据特点，提供了大尺幅数据切片与拼接，支持读取`tif`、`png`、 `jpeg`、 `bmp`、 `img`以及 `npy`等格式，支持地理信息保存和超分辨率。

* <img src="./docs/images/f2.png" width="20"/> **覆盖任务广**：支持目标检测、图像分割、变化检测、参数反演等多种任务

* <img src="./docs/images/f3.png" width="20"/> **高性能**：支持多进程异步I/O、多卡并行训练、评估等加速策略，结合飞桨核心框架的显存优化功能，可大幅度减少分割模型的训练开销，让开发者更低成本、更高效地完成图像遥感图像的开发和训练。

## 产品矩阵<img src="./docs/images/model.png" width="30"/>

<table align="center">
  <tbody>
    <tr align="center" valign="bottom">
      <td>
        <b>模型总览</b>
      </td>
      <td>
        <b>数据增强</b>
      </td>
      <td>
        <b>遥感工具</b>
      </td>
      <td>
        <b>实践案例</b>
      </td>
    </tr>
    <tr valign="top">
      <td>
        <b>场景分类</b><br>
        <ul>
          <li>ResNet</li>
          <li>MobileNet</li>
          <li>PPLCNet</li>
        </ul>
        <b>语义分割</b><br>
        <ul>
          <li>OCRNet</li>
          <li>FarSeg</li>
          <li>SegFormer</li>
        </ul>
        <b>目标检测</b><br>
        <ul>
          <li>PP-YOLO</li>
          <li>Faster RCNN</li>
          <li>FCOS</li>
        </ul>
        <b>超分/去噪</b><br>
        <ul>
          <li>RealSR</li>
          <li>ESRGAN</li>
          <li>LESRCNN</li>
        </ul>
        <b>变化检测</b><br>
        <ul>
          <li>DSIFN</li>
          <li>STANet</li>
          <li>UNetSiamDiff</li>
        </ul>
      </td>
      <td>
        <b>数据增强</b><br>
        <ul>
          <li>Resize</li>  
          <li>RandomResize</li>  
          <li>ResizeByShort</li>
          <li>RandomResizeByShort</li>
          <li>ResizeByLong</li>  
          <li>RandomHorizontalFlip</li>  
          <li>RandomVerticalFlip</li>
          <li>Normalize</li>
          <li>CenterCrop</li>
          <li>RandomCrop</li>
          <li>RandomScaleAspect</li>  
          <li>RandomExpand</li>
          <li>Padding</li>
          <li>MixupImage</li>  
          <li>RandomDistort</li>  
          <li>RandomBlur</li>  
          <li>Defogging</li>  
          <li>DimReducing</li>  
          <li>BandSelecting</li>  
          <li>RandomSwap</li>
        </ul>  
      </td>
      <td>
        <b>数据格式转换</b><br>
        <ul>
          <li>geojson to mask</li>
          <li>mask to shpfile</li>
        </ul>
        <b>数据预处理</b><br>
        <ul>
          <li>data split</li>
          <li>images match</li>
          <li>bands select</li>
        </ul>
      </td>
      <td>
        <b>遥感场景分类</b><br>
        <ul>
          <li>待更</li>
        </ul>
        <b>遥感语义分割</b><br>
        <ul>
          <li>待更</li>
        </ul>
        <b>遥感目标检测</b><br>
        <ul>
          <li>待更</li>
        </ul>
        <b>遥感变化检测</b><br>
        <ul>
          <li>待更</li>
        </ul>
        <b>遥感影像超分</b><br>
        <ul>
          <li>待更</li>
        </ul>
      </td>  
    </tr>
  </tbody>
</table>


### 代码结构

这部分将展示PaddleRS的文件结构全貌。文件树如下：

```
├── deploy               # 部署相关的文档和脚本
├── docs                 # 整个项目文档及图片
├── paddlers  
│     ├── custom_models  # 自定义网络模型代码
│     ├── datasets       # 数据加载相关代码
│     ├── models         # 套件网络模型代码
│     ├── tasks          # 相关任务代码
│     ├── tools          # 相关脚本
│     ├── transforms     # 数据处理及增强相关代码
│     └── utils          # 各种实用程序文件
├── tools                # 用于处理遥感数据的脚本
└── tutorials
      └── train          # 训练教程
```

## 技术交流 <img src="./docs/images/chat.png" width="30"/>

* 如果你发现任何PaddleRS存在的问题或者是建议, 欢迎通过[GitHub Issues](https://github.com/PaddleCV-SIG/PaddleRS/issues)给我们提issues。
* 欢迎加入PaddleRS 微信群
<div align="center">
<img src="./docs/images/wechat.jpg"  width = "200" />  
</div>

## 使用教程 <img src="./docs/images/teach.png" width="30"/>

* [快速上手PaddleRS](./tutorials/train/README.md)
* 准备数据集
   * [遥感数据介绍](./docs/data/rs_data_cn.md)
   * [遥感数据集](./docs/data/dataset_cn.md)
   * [智能标注工具EISeg](https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.4/EISeg)
   * [遥感数据处理脚本](./docs/data/tools.md)
* APIs
   * [模型库](./docs/apis/model_zoo.md)
   * [数据增强](./docs/apis/transforms.md)
* 模型训练
* 推理部署
* 应用案例
  * [【官方】第十一届 “中国软件杯”大学生软件设计大赛—A组：变化检测](https://aistudio.baidu.com/aistudio/projectdetail/3684588)
  * [使用超分模块提高真实的低分辨率无人机影像的分割精度](https://aistudio.baidu.com/aistudio/projectdetail/3696814)

## 许可证书

本项目的发布受Apache 2.0 license许可认证。

## 贡献说明 <img src="./docs/images/love.png" width="30"/>

我们非常欢迎你可以为PaddleRS提供代码，也十分感谢你的反馈。代码注释规范请参考[PaddleRS代码注释规范](https://github.com/PaddleCV-SIG/PaddleRS/wiki/PaddleRS代码注释规范)。

## 学术引用 <img src="./docs/images/yinyong.png" width="30"/>

如果我们的项目在学术上帮助到你，请考虑以下引用：

```latex
@misc{paddlers2022,
    title={PaddleRS, Awesome Remote Sensing Toolkit based on PaddlePaddle},
    author={PaddlePaddle Authors},
    howpublished = {\url{https://github.com/PaddleCV-SIG/PaddleRS}},
    year={2022}
}
```
