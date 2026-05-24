简体中文 | [English](quick_start_en.md)

# 快速开始

## 环境准备

1. [安装PaddlePaddle](https://www.paddlepaddle.org.cn/install/quick)
  - 版本要求：PaddlePaddle>=2.5.0

2. 安装PaddleRS

如果希望获取更加稳定的体验，请下载安装[PaddleRS发行版](https://github.com/PaddlePaddle/PaddleRS/releases)。

```shell
pip install .
```

PaddleRS代码会跟随开发进度不断更新，如果希望使用最新功能，请安装PaddleRS develop分支。安装方式如下：

```shell
git clone https://github.com/PaddlePaddle/PaddleRS
cd PaddleRS
git checkout develop
pip install .
```

若在执行`pip install .`时下载依赖缓慢或超时，可以在`setup.py`相同目录下新建`setup.cfg`，并输入以下内容，则可通过清华源进行加速下载：

```
[easy_install]
index-url=https://pypi.tuna.tsinghua.edu.cn/simple
```

3. （可选）安装GDAL

PaddleRS支持对多种类型卫星数据的读取。完整使用PaddleRS的遥感数据读取功能需要安装GDAL，安装方式如下：

  - Linux / MacOS

推荐使用conda进行安装:

```shell
conda install gdal
```

  - Windows

Windows用户可以在[此站点](https://www.lfd.uci.edu/~gohlke/pythonlibs/#gdal)下载与Python和系统版本相对应的.whl格式安装包到本地，以*GDAL‑3.3.3‑cp39‑cp39‑win_amd64.whl*为例，使用pip工具安装:

```shell
pip install GDAL‑3.3.3‑cp39‑cp39‑win_amd64.whl
```

4. （可选）安装ext_op

PaddleRS支持旋转目标检测，在使用之前需要安装自定义外部算子库`ext_op`，安装方式如下：

```shell
cd paddlers/models/ppdet/ext_op
python setup.py install
```


除了采用上述安装步骤以外，PaddleRS也提供Docker安装方式，具体请参考[文档](./docker_cn.md)。

## 模型训练

+ 在安装完成PaddleRS后，即可开始模型训练。
+ 模型训练可参考：[使用教程——训练模型](../tutorials/train/README_CN.md)

## 模型精度验证

模型训练完成后，需要对模型进行精度验证，以确保模型的预测效果符合预期。以DeepLab V3+图像分割模型为例，可以使用以下命令启动：

```python
import paddlers as pdrs
from paddlers import transforms as T

# 加载模型
model = pdrs.load_model('output/deeplabv3p/best_model')

# 组合数据变换算子
eval_transforms = [
    T.Resize(target_size=512),
    T.Normalize(
        mean=[0.5] * NUM_BANDS, std=[0.5] * NUM_BANDS),
    T.ReloadMask()
]

# 加载验证集
dataset = pdrs.datasets.SegDataset(
    data_dir='dataset',
    file_list='dataset/val/list.txt',
    label_list='dataset/labels.txt',
    transforms=eval_transforms)

# 进行验证
result = model.evaluate(dataset)

print(result)
```

## 模型部署

### 模型导出

模型导出可参考：[部署模型导出](../deploy/export/README.md)

### Python部署

Python部署可参考：[Python部署](../deploy/README.md)
