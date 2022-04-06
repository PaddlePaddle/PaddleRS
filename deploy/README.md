# Python部署

PaddleRS已经集成了基于Python的高性能预测（prediction）接口。在安装PaddleRS后，可参照如下代码示例执行预测。

## 部署模型导出

在服务端部署模型时需要首先将训练过程中保存的模型导出为部署格式，具体的导出步骤请参考文档[部署模型导出](/deploy/export/README.md)。

## 预测接口调用

* **基本使用**

以下是一个调用PaddleRS Python预测接口的实例。首先构建`Predictor`对象，然后调用`Predictor`的`predict()`方法执行预测。

```python
import paddlers as pdrs
# 将导出模型所在目录传入Predictor的构造方法中
predictor = pdrs.deploy.Predictor('./inference_model')
# img_file参数指定输入图像路径
result = predictor.predict(img_file='test.jpg')
```

* **在预测过程中评估模型预测速度**

加载模型后，对前几张图片的预测速度会较慢，这是因为程序刚启动时需要进行内存、显存初始化等步骤。通常，在处理20-30张图片后，模型的预测速度能够达到稳定值。基于这一观察，**如果需要评估模型的预测速度，可通过指定预热轮数`warmup_iters`对模型进行预热**。此外，**为获得更加精准的预测速度估计值，可指定重复`repeats`次预测后计算平均耗时**。

```python
import paddlers as pdrs
predictor = pdrs.deploy.Predictor('./inference_model')
result = predictor.predict(img_file='test.jpg',
                           warmup_iters=100,
                           repeats=100)
```
