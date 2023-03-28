# 快速开始

## 环境准备

环境准备可参考：[使用教程——环境准备](../tutorials/train/README.md)

## 模型训练

+ 在安装完成PaddleRS后，即可开始模型训练。
+ 模型训练可参考：[使用教程——训练模型](../tutorials/train/README.md)

## 模型精度验证

模型训练完成后，需要对模型进行精度验证，以确保模型的预测效果符合预期。以DeepLab V3+图像分割模型为例，可以使用以下命令启动：

```python
import paddlex as pdx

# 加载模型
model = pdx.load_model('output/deeplabv3p/best_model')

# 加载验证集
dataset = pdx.datasets.SegDataset(
    data_dir='dataset/val',
    file_list='dataset/val/list.txt',
    label_list='dataset/labels.txt',
    transforms=model.eval_transforms)

# 进行验证
result = model.evaluate(dataset, batch_size=1, epoch_id=None, return_details=True)

print(result)
```

在上述代码中，`pdx.load_model()`方法用于加载预训练的DeepLabV3P模型，`pdx.datasets.SegDataset()`方法用于加载验证集数据。`model.evaluate()`方法接受验证集数据集、批大小和轮数等参数，并返回包括预测结果和指标评估在内的验证结果。最后，我们可以打印输出验证结果。


## 模型部署

### 模型导出

模型导出可参考：[部署模型导出](../deploy/export/README.md)

### Python部署

python部署可参考：[Python部署](../deploy/README.md)
