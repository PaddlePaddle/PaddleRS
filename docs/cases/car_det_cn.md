# [PaddleRS：无人机汽车识别](https://aistudio.baidu.com/aistudio/projectdetail/3713122)

基于0.5m的高分辨率无人机影像，我们希望能够使用目标检测的方法找到影像中的汽车。项目将基于PaddleRS完成该任务。

## 1 数据准备

数据来自于[DFC2018 Houston](https://hyperspectral.ee.uh.edu/?page_id=1075)，裁剪为1400张596x601大小的图块，由手工标注而成并按照9:1划分训练集和数据集。

```python
# 解压数据集
! mkdir -p dataset
! unzip -oq data/data56250/carDetection_RGB.zip -d dataset
```

```python
# 划分数据集
import os
import os.path as osp
import random

def get_data_list(data_dir):
    random.seed(666)
    mode = ["train_list", "val_list"]
    dir_path = osp.join(data_dir, "JPEGImages")
    files = [f.split(".")[0] for f in os.listdir(dir_path)]
    random.shuffle(files)  # 打乱顺序
    with open(osp.join(data_dir, f"{mode[0]}.txt"), "w") as f_tr:
        with open(osp.join(data_dir, f"{mode[1]}.txt"), "w") as f_va:
            for i, name in enumerate(files):
                if (i % 10) == 0:  # 训练集与测试集为9:1
                    f_va.write(f"JPEGImages/{name}.jpg Annotations/{name}.xml\n")
                else:
                    f_tr.write(f"JPEGImages/{name}.jpg Annotations/{name}.xml\n")
    labels = ["car"]
    txt_str = "\n".join(labels)
    with open((data_dir + "/" + f"label_list.txt"), "w") as f:
        f.write(txt_str)
    print("Finished!")

get_data_list("dataset")
```

## 2 PaddleRS准备

PaddleRS是基于飞桨开发的遥感处理平台，支持遥感图像分类，目标检测，图像分割，以及变化检测等常用遥感任务，帮助开发者更便捷地完成从训练到部署全流程遥感深度学习应用。

github：[https://github.com/PaddleCV-SIG/PaddleRS](https://github.com/PaddleCV-SIG/PaddleRS)

```python
! git clone https://github.com/PaddleCV-SIG/PaddleRS.git
! pip install -q -r PaddleRS/requirements.txt

import sys
sys.path.append("PaddleRS")
```

## 3 模型训练

PaddleRS借鉴PaddleSeg的API设计模式并进行了较高程度的封装，可以方便的完成数据、模型等的定义，快速开始模型的训练迭代。

### 3.1 数据定义

主要通过`datasets`和`transforms`两个组件完成任务，`datasets`中有包含分割检测分类等多任务的数据加载API，而`transforms`集成了大部分通用或单独的数据增强API，目前可以通过源码查看。

```python
import os
import os.path as osp
from paddlers.datasets import VOCDetection
from paddlers import transforms as T

# 定义数据增强
train_transforms = T.Compose([
    T.RandomDistort(),
    T.RandomCrop(),
    T.RandomHorizontalFlip(),
    T.BatchRandomResize(
        target_sizes=[512, 544, 576, 608, 640, 672, 704],
        interp='RANDOM'),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
eval_transforms = T.Compose([
    T.Resize(target_size=608, interp='CUBIC'),
    T.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 定义数据集
data_dir = "dataset"
train_file_list = osp.join(data_dir, 'train_list.txt')
val_file_list = osp.join(data_dir, 'val_list.txt')
label_file_list = osp.join(data_dir, 'label_list.txt')
train_dataset = VOCDetection(
    data_dir=data_dir,
    file_list=train_file_list,
    label_list=label_file_list,
    transforms=train_transforms,
    shuffle=True)
eval_dataset = VOCDetection(
    data_dir=data_dir,
    file_list=train_file_list,
    label_list=label_file_list,
    transforms=eval_transforms,
    shuffle=False)
```

### 3.2 模型准备

PaddleRS将模型分别放置于`models`和`custom_models`中，分别包含了Paddle四大套件的模型结构以及与遥感、变化检测等相关的模型结构。通过`tasks`进行了模型的封装，集成了Loss、Opt、Metrics等，可根据需要进行修改。这里以默认的PPYOLOv2为例。

```python
from paddlers.tasks.object_detector import PPYOLOv2

num_classes = len(train_dataset.labels)
model = PPYOLOv2(num_classes=num_classes)
```

```python
model.train(
    num_epochs=30,
    train_dataset=train_dataset,
    train_batch_size=16,
    eval_dataset=eval_dataset,
    pretrain_weights="COCO",
    learning_rate=3e-5,
    warmup_steps=10,
    warmup_start_lr=0.0,
    save_interval_epochs=5,
    lr_decay_epochs=[10, 20],
    save_dir="output",
    use_vdl=True)
```

## 4 模型评估

只需要调用evaluate即可完成预测。

```python
model.evaluate(eval_dataset)
```

返回如下输出。

```
    2022-03-30 19:59:13 [INFO]    Start to evaluate(total_samples=944, total_steps=944)...
    2022-03-30 20:00:05 [INFO]    Accumulating evaluatation results...

    OrderedDict([('bbox_map', 90.33284968764544)])
```

## 5 模型预测

PaddleRS的目标检测task可以方便的给出坐标、类别和分数，可供自行进行一些后处理。也可以直接使用visualize_detection进行可视化。下面对一张测试图像进行预测并可视化。

```python
from paddlers.tasks.utils.visualize import visualize_detection
import matplotlib.pyplot as plt

%matplotlib inline

img_path = "dataset/JPEGImages/UH_NAD83_272056_3289689_58.jpg"
pred = model.predict(img_path, eval_transforms)
vis_img = visualize_detection(img_path, pred, save_dir=None)
plt.figure(figsize=(10, 10))
plt.imshow(vis_img)
plt.show()
```

![output_13_0](https://user-images.githubusercontent.com/71769312/161358212-5f525ba3-059c-4c07-9d2e-ed4334069983.png)

## 总结

- 这里PPYOLOv2的效果很不错，后续在目标检测方面，将会为PaddleRS增加滑框预测以及GeoJSON等数据格式的导出。
