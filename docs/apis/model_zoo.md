# 模型库

PaddleRS的基础模型库来自[PaddleClas](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.3/docs/zh_CN/algorithm_introduction/ImageNet_models.md)、[PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.4/docs/model_zoo_overview_cn.md)、[PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.3/README_cn.md#模型库)以及[PaddleGAN](https://github.com/PaddlePaddle/PaddleGAN/blob/develop/README_cn.md#模型库)，可以通过相关的链接进行查看。而在此之外，PaddleRS也针对遥感任务添加了一些特有的模型库，可用于遥感图像语义分割、遥感变化检测等。

## 自定义模型库

| 模型名称        | 用途     | 
| --------------- | -------- | 
| FarSeg          | 语义分割 |
| BIT             | 变化检测 |
| CDNet           | 变化检测 |
| DSIFN           | 变化检测 |
| STANet          | 变化检测 |
| SNUNet          | 变化检测 | 
| DSAMNet         | 变化检测 |
| FCEarlyFusion | 变化检测 | 
| FCSiamConc    | 变化检测 | 
| FCSiamDiff    | 变化检测 | 


## 如何导入

模型均位于`paddlers/models`和`paddlers/custom_models`中，对于套件中的模型可以通过如下方法进行使用

```python
from paddlers.models import xxxx
```

而PaddleRS所特有的模型可以通过如下方法调用

```python
from paddlers.custom_models import xxxx
```
