# PaddleRS

<div align="center">

<p align="center">
  <img src="./docs/images/logo.png" align="middle" width = "500" />
</p>

**飞桨高性能遥感图像处理开发套件，端到端地完成从训练到部署的全流程遥感深度学习应用。**

<!-- [![Build Status](https://travis-ci.org/PaddlePaddle/PaddleSeg.svg?branch=release/2.1)](https://travis-ci.org/PaddlePaddle/PaddleSeg) -->
<!-- [![Version](https://img.shields.io/github/release/PaddlePaddle/PaddleSeg.svg)](https://github.com/PaddlePaddle/PaddleSeg/releases) -->
[![License](https://img.shields.io/badge/license-Apache%202-blue.svg)](LICENSE)
![python version](https://img.shields.io/badge/python-3.6+-orange.svg)
![support os](https://img.shields.io/badge/os-linux%2C%20win%2C%20mac-yellow.svg)
</div>



## 最新动态 <img src="./docs/images/seg_news_icon.png" width="40"/>

* [2022-03-30] PaddleRS alpha版本发布！详细发版信息请参考[Release Note](https://github.com/PaddleCV-SIG)。


## 简介
PaddleRS是xxx、xxx、xxx等遥感科研院所共同基于飞桨开发的遥感处理平台，支持遥感图像分类，目标检测，图像分割，以及变化检测等常用遥感任务，帮助开发者更便捷地完成从训练到部署全流程遥感深度学习应用。


<div align="center">
<img src="docs/images/whole_image.png"  width = "2000" />  
</div>


----------------
## 特性 <img src="./docs/images/feature.png" width="30"/>


* <img src="./docs/images/f1.png" width="20"/> **特有的遥感数据处理模块**：针对遥感行业数据特点，提供了大尺幅数据切片与拼接，支持读取`tif`, `png`, `jpeg`, `bmp`, `img`, `npy`.
等格式，支持地理信息保存和超分辨率。

* <img src="./docs/images/f2.png" width="20"/> **覆盖任务广**：支持目标检测、图像分割、变化检测、参数反演等多种任务

* <img src="./docs/images/f3.png" width="20"/> **高性能**：支持多进程异步I/O、多卡并行训练、评估等加速策略，结合飞桨核心框架的显存优化功能，可大幅度减少分割模型的训练开销，让开发者更低成本、更高效地完成图像遥感图像的开发和训练。

----------

## 技术交流 <img src="./docs/images/chat.png" width="30"/>

* 如果你发现任何PaddleRS存在的问题或者是建议, 欢迎通过[GitHub Issues](https://github.com/PaddleCV-SIG/PaddleRS/issues)给我们提issues。
* 欢迎加入PaddleRS 微信群
<div align="center">
<img src="./docs/images/wechat.png"  width = "200" />  
</div>




## 使用教程 <img src="./docs/images/teach.png" width="30"/>

* [环境安装](./docs/install_cn.md)
* [快速上手PaddleRS](./docs/whole_process_cn.md)
*  准备数据集
   * [数据集说明](./docs/data/marker/marker_cn.md)
   * [智能标注工具EISeg](./docs/data/transform/transform_cn.md)

* [模型训练与评估](/docs/train/train_cn.md)
* [预测与可视化](./docs/predict/predict_cn.md)

* 模型导出
    * [导出预测模型](./docs/model_export_cn.md)
    * [导出ONNX模型](./docs/model_export_onnx_cn.md)

* 模型压缩
    * [量化](./docs/slim/quant/quant_cn.md)
    * [蒸馏](./docs/slim/distill/distill_cn.md)
    * [裁剪](./docs/slim/prune/prune_cn.md)

* 模型部署
    * [Paddle Inference部署(Python)](./docs/deployment/inference/python_inference_cn.md)
    * [Paddle Inference部署(C++)](./docs/deployment/inference/cpp_inference_cn.md)
    * [Paddle Lite部署](./docs/deployment/lite/lite_cn.md)



*  API使用教程
    * [API文档说明](./docs/apis/README_CN.md)
    * [API应用案例](./docs/api_example_cn.md)
*  重要模块说明
    * [数据增强](./docs/module/data/data_cn.md)
    * [Loss说明](./docs/module/loss/losses_cn.md)
*  二次开发教程
    * [配置文件详解](./docs/design/use/use_cn.md)
    * [如何创造自己的模型](./docs/design/create/add_new_model_cn.md)
*  模型贡献
    * [提交PR说明](./docs/pr/pr/pr_cn.md)
    * [模型PR规范](./docs/pr/pr/style_cn.md)

* [常见问题汇总](./docs/faq/faq/faq_cn.md)

## 实践案例 <img src="./docs/images/anli.png" width="20"/>

- [地块分割](./EISeg)
- [舰船检测](./contrib/Matting)
- [农作物增长预测](./contrib/PP-HumanSeg)
- [城区变化检测](./contrib/CityscapesSOTA)



## 许可证书
本项目的发布受Apache 2.0 license许可认证。

## 贡献说明 <img src="./docs/images/love.png" width="20"/>
本项目的发布受Apache 2.0 license许可认证。


## 学术引用 <img src="./docs/images/yinyong.png" width="30"/>

如果我们的项目在学术上帮助到你，请考虑以下引用：

```latex
@misc{liu2021paddleseg,
      title={PaddleSeg: A High-Efficient Development Toolkit for Image Segmentation},
      author={Yi Liu and Lutao Chu and Guowei Chen and Zewu Wu and Zeyu Chen and Baohua Lai and Yuying Hao},
      year={2021},
      eprint={2101.06175},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

@misc{paddleseg2019,
    title={PaddleSeg, End-to-end image segmentation kit based on PaddlePaddle},
    author={PaddlePaddle Authors},
    howpublished = {\url{https://github.com/PaddlePaddle/PaddleSeg}},
    year={2019}
}
```
