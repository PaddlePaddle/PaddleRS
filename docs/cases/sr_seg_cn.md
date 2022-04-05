# [使用图像超分提高低分辨率无人机影像的分割精度](https://aistudio.baidu.com/aistudio/projectdetail/3696814)

## 一、项目背景

- 前段时间写了个项目：[PaddleSeg：使用Transfomer模型对航空遥感图像分割](https://aistudio.baidu.com/aistudio/projectdetail/3565870)，项目利用PaddleSeg模块训练Transfomer类的语义分割模型，在UDD6数据集中**mIOU达到74.50%** ，原论文使用DeepLabV3+的mIOU为73.18%， **高1.32%** ，训练效果图如下，其中：车辆：红色；道路：浅蓝色；植被：深蓝色；建筑立面：亮绿色；建筑屋顶：紫色；其他：焦绿色

```python
%cd /home/aistudio/
import matplotlib.pyplot as plt
from PIL import Image

output = Image.open(r"work/example/Seg/UDD6_result/added_prediction/000161.JPG")

plt.figure(figsize=(18, 12))  # 设置窗口大小
plt.imshow(output), plt.axis('off')
```

![output_1_2](https://user-images.githubusercontent.com/71769312/161358238-5dc85c26-de33-4552-83ea-ad9936a5c85a.png)

- 训练的结果很不错，所使用的UDD6数据是从北京、葫芦岛、沧州、郑州四个城市，使用大疆精灵四无人机在60m-100m高度之间采集。但是，**在实际的生产过程中，城市、飞行的高度、图像的质量会发生变化**
- 采集飞行高度升高可以在相同时间内获取更大面积的数据，但分辨率会降低，对低质量的数据，**直接使用先前训练的数据预测效果不理想**，**再标注数据、训练模型将是一个不小的工作量**，解决的方法除了提升模型的泛化能力，也可以考虑使用图像超分对低质量的无人机图像重建，然后再进行预测
- 本项目使用PaddleRS提供的无人机遥感图像超分模块，对**真实的低质量无人机影像**数据进行**超分**，然后再使用前段时间用UDD6训练的Segformer模型预测，与直接使用低分辨率模型对比。由于没有对低质量数据进行标注无法计算指标。但人眼判别，超分之后的预测结果更好，**左边是人工标注的label，中间是低分辨率的预测结果，右边是超分辨率重建后的结果**

```python
img = Image.open(r"work/example/Seg/gt_result/data_05_2_14.png")
lq = Image.open(r"work/example/Seg/lq_result/added_prediction/data_05_2_14.png")
sr = Image.open(r"work/example/Seg/sr_result/added_prediction/data_05_2_14.png")

plt.figure(figsize=(18, 12))
plt.subplot(1,3,1), plt.title('GT')
plt.imshow(img), plt.axis('off')
plt.subplot(1,3,2), plt.title('predict_LR')
plt.imshow(lq), plt.axis('off')
plt.subplot(1,3,3), plt.title('predict_SR')
plt.imshow(sr), plt.axis('off')
plt.show()
```

![output_3_0](https://user-images.githubusercontent.com/71769312/161358300-b85cdda4-7d1f-40e7-a39b-74b2cd5347b6.png)

## 二、数据介绍与展示
- 使用的数据是使用大疆精灵四无人机在**上海，飞行高度为300m**采集的，采集的时候天气也一般，可以看后续的示例发现质量不高。由于只是展示超分重建后进行预测的效果，所以只是简单标注了其中5张照片，毕竟**标注数据真的是一件很费力的事！** 要是能用公开数据集训练的模型来预测自己的数据，这多是一件美事！
- 部分标注数据展示如下

```python
add_lb = Image.open(r"work/example/Seg/gt_result/data_05_2_19.png")
lb = Image.open(r"work/example/Seg/gt_label/data_05_2_19.png")
img = Image.open(r"work/ValData/DJI300/data_05_2_19.png")

plt.figure(figsize=(18, 12))
plt.subplot(1,3,1), plt.title('image')
plt.imshow(img), plt.axis('off')
plt.subplot(1,3,2), plt.title('label')
plt.imshow(lb), plt.axis('off')
plt.subplot(1,3,3), plt.title('add_label')
plt.imshow(add_lb), plt.axis('off')
plt.show()
```

![output_5_0](https://user-images.githubusercontent.com/71769312/161358312-3c16cbb0-1162-4fbe-b3d6-9403502aefef.png)

## 三、无人机遥感图像超分
- 因为PaddleRS提供了预训练的超分模型，所以这步主要分为以下两个步骤：
    - 准备PaddleRS并设置好环境
    - 调用PaddleRS中的超分预测接口，对低分辨率无人机影像进行**超分重建**

```python
# 从github上克隆仓库
!git clone https://github.com/PaddleCV-SIG/PaddleRS.git
```

```python
# 安装依赖，大概一分多钟
%cd PaddleRS/
!pip install -r requirements.txt
```

```python
# 进行图像超分处理，使用的模型为DRN
import os
import paddle
import numpy as np
from PIL import Image
from paddlers.models.ppgan.apps.drn_predictor import DRNPredictor

# 输出预测结果的文件夹
output = r'../work/example'
# 待输入的低分辨率影像位置
input_dir = r"../work/ValData/DJI300"

paddle.device.set_device("gpu:0")  # 若是cpu环境，则替换为 paddle.device.set_device("cpu")
predictor = DRNPredictor(output)  # 实例化

filenames = [f for f in os.listdir(input_dir) if f.endswith('.png')]
for filename in filenames:
    imgPath = os.path.join(input_dir, filename)  
    predictor.run(imgPath)  # 预测
```

- 超分重建结果前后对比展示

```python
# 可视化
import os
import matplotlib.pyplot as plt
%matplotlib inline

lq_dir = r"../work/ValData/DJI300"  # 低分辨率影像文件夹
sr_dir = r"../work/example/DRN"  # 超分辨率影像所在文件夹
img_list = [f for f in os.listdir(lq_dir) if f.endswith('.png')]
show_num = 3  # 展示多少对影像
for i in range(show_num):
    lq_box = (100, 100, 175, 175)
    sr_box = (400, 400, 700, 700)
    filename = img_list[i]
    image = Image.open(os.path.join(lq_dir, filename)).crop(lq_box)  # 读取低分辨率影像
    sr_img = Image.open(os.path.join(sr_dir, filename)).crop(sr_box)  # 读取超分辨率影像

    plt.figure(figsize=(12, 8))
    plt.subplot(1,2,1), plt.title('Input')
    plt.imshow(image), plt.axis('off')
    plt.subplot(1,2,2), plt.title('Output')
    plt.imshow(sr_img), plt.axis('off')
    plt.show()
```

![output_11_0](https://user-images.githubusercontent.com/71769312/161358324-c45d750d-b47e-4201-b70c-3c374498fd86.png)

![output_11_1](https://user-images.githubusercontent.com/71769312/161358335-0b85035e-0a9d-4b5a-8d0c-14ecaeffd947.png)

![output_11_2](https://user-images.githubusercontent.com/71769312/161358342-d2875098-cb9b-4bc2-99b0-bcab4c1bc5e1.png)

## 四、超分前后图像分割效果对比

- 使用的模型为Segformer_b3,用UDD6数据集训练了40000次
- 已经将性能最好的模型以及.yml文件放在work文件夹下
- 运行以下命令可对指定的文件夹下的图像进行预测
- 首先用该模型对低质量的无人机数据进行预测，然后再使用超分重建后的图像预测，最后对比一下预测的效果

```python
%cd ..
# clone PaddleSeg的项目
!git clone https://gitee.com/paddlepaddle/PaddleSeg
```

```python
# 安装依赖
%cd /home/aistudio/PaddleSeg
!pip install  -r requirements.txt
```

```python
# 对低分辨率的无人机影像进行预测
!python predict.py \
       --config ../work/segformer_b3_UDD.yml \
       --model_path ../work/best_model/model.pdparams \
       --image_path ../work/ValData/DJI300 \
       --save_dir ../work/example/Seg/lq_result
```

```python
# 对使用DRN超分重建后的影像进行预测
!python predict.py \
       --config ../work/segformer_b3_UDD.yml \
       --model_path ../work/best_model/model.pdparams \
       --image_path ../work/example/DRN \
       --save_dir ../work/example/Seg/sr_result
```

**展示预测结果**
- 其中，颜色如下：

|   种类 | 颜色   |
|----------|---------|
|  **其他**  |  焦绿色  |
| 建筑外立面 |  亮绿色  |
|  **道路**  |  淡蓝色  |
| 植被 |  深蓝色  |
|  **车辆** |  红色  |
| 屋顶 |  紫色  |

- 由于只标注了五张图片，所以只展示五张图片的结果，剩下的预测结果均在 `work/example/Seg/`文件夹下,其中左边是真值，中间是低分辨率影像预测结果，右边是超分重建后预测结果

```python
# 展示部分预测的结果
%cd /home/aistudio/
import matplotlib.pyplot as plt
from PIL import Image
import os

img_dir = r"work/example/Seg/gt_result"  # 低分辨率影像文件夹
lq_dir = r"work/example/Seg/lq_result/added_prediction"
sr_dir = r"work/example/Seg/sr_result/added_prediction"  # 超分辨率预测的结果影像所在文件夹
img_list = [f for f in os.listdir(img_dir) if f.endswith('.png') ]
for filename in img_list:
    img = Image.open(os.path.join(img_dir, filename))
    lq_pred = Image.open(os.path.join(lq_dir, filename))
    sr_pred = Image.open(os.path.join(sr_dir, filename))

    plt.figure(figsize=(12, 8))
    plt.subplot(1,3,1), plt.title('GT')
    plt.imshow(img), plt.axis('off')
    plt.subplot(1,3,2), plt.title('LR_pred')
    plt.imshow(lq_pred), plt.axis('off')
    plt.subplot(1,3,3), plt.title('SR_pred')
    plt.imshow(sr_pred), plt.axis('off')
    plt.show()

```

![output_18_1](https://user-images.githubusercontent.com/71769312/161358523-42063419-b490-4fca-b0d4-cb2b05f7f74a.png)

![output_18_2](https://user-images.githubusercontent.com/71769312/161358556-e2f66be4-4758-4c7a-9b3b-636aa2b53215.png)

![output_18_3](https://user-images.githubusercontent.com/71769312/161358599-e74696f3-b374-4d5c-a9f4-7ffaef8938a0.png)

![output_18_4](https://user-images.githubusercontent.com/71769312/161358621-c3c0d225-b67f-4bff-91ba-4be714162584.png)

![output_18_5](https://user-images.githubusercontent.com/71769312/161358643-9aba7db1-6c68-48f2-be53-8eec30f27d60.png)

## 五、总结
- 本项目调用PaddleRS提供的超分重建接口，选用DRN模型对真实采集的低分辨率影像进行重建，再对重建后的图像进行分割，从结果上看，**超分重建后的图片的分割结果更好**
- **不足之处**：虽然相对于低分辨率影像，超分重建后的预测精度从目视的角度有所提高，但是并没有达到UDD6测试集中的效果，所以**模型的泛化能力也需要提高才行，光靠超分重建依然不够**
- **后续工作**：将会把超分重建这一步整合到PaddleRS中的transform模块，在high-level任务预测之前可以进行调用改善图像质量，请大家多多关注[PaddleRS](https://github.com/PaddleCV-SIG/PaddleRS)
