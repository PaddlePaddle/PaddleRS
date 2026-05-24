# 建筑物提取全流程案例

PaddleRS提供遥感模型训练和推理能力，结合EISeg提供的标注能力和GeoView提供的部署与展示能力，即可全流程地完成遥感图像分割任务。本案例基于Docker环境，在Windows 10系统使用上述工具实现对卫星影像建筑物提取任务从标注到训练再到部署的全流程开发。

## 〇、准备

- 构建PaddleRS基础Docker镜像，详细过程请参考PaddleRS关于Docker镜像构建的[文档](https://github.com/PaddlePaddle/PaddleRS/blob/release/1.1/docs/docker_cn.md)。
- 为使用[GeoView](https://github.com/PaddleCV-SIG/GeoView)提供的遥感影像智能解译功能，可在PaddleRS基础镜像基础上使用本案例提供的`Dockerfile`构建GeoView镜像。若PaddleRS基础镜像名称为`<imageName>`，需要将`Dockerfile`中的`FROM paddlers:latest`改为`FROM <imageName>`，然后构建镜像：

```shell
docker build -t geoview:latest -f Dockerfile .
```

基于构建的镜像创建并运行Docker容器。

```shell
docker run -it -v <本机文件夹绝对路径>:<容器文件夹绝对路径> [--gpus all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility -e NVIDIA_VISIBLE_DEVICES=all] geoview:latest /bin/bash
```

`-v`选项指定的`<本机文件夹绝对路径>`可用于在Docker容器和宿主机之间共享文件。

为便于说明，本案例提供一幅青岛地区tif影像作为示例数据。在Docker容器中执行如下指令下载数据并解压到`/usr/qingdao`目录：

```shell
wget https://paddlers.bj.bcebos.com/datasets/qingdao.zip
unzip -d /usr/qingdao qingdao.zip
```

## 一、数据标注

- 切换到`/opt/GeoView/PaddleRS`目录，安装PaddleRS。

```shell
python setup.py install
```

- 切分图像。虽然EISeg可以直接读取大图并进行分块标注和保存，但为了控制标注的数量，可以先使用PaddleRS提供的工具预先对图像进行切分。

```shell
cd tools/
python split.py --image_path /usr/qingdao/qingdao.tif --block_size 512 --save_dir /usr/qingdao/dataset/
```

- 等待进度条完成后数据划分完毕。将[适用于建筑物提取的交互式分割模型参数](https://paddleseg.bj.bcebos.com/eiseg/0.4/static_hrnet18_ocr48_rsbuilding_instance.zip)下载到共享文件夹中，打开[VcXsrv](https://sourceforge.net/projects/vcxsrv/)（宿主机系统为Windows 10），准备使用EISeg进行标注。具体操作请参考PaddleRS [Docker镜像构建与使用文档](https://github.com/PaddlePaddle/PaddleRS/blob/release/1.1/docs/docker_cn.md#2-%E9%95%9C%E5%83%8F%E4%BD%BF%E7%94%A8)中关于EISeg使用的部分。

![eiseg](https://user-images.githubusercontent.com/71769312/222040539-34a369f3-6da8-4047-a3a5-ebf9b831d175.png)

- 加载模型和数据后标注情况如下。关于EISeg的使用方法请参考[这里](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.8/EISeg/docs/image.md)。

![eiseg_labeling](https://user-images.githubusercontent.com/71769312/222041481-da2398e4-b312-418f-9cf7-e2c22badfe8a.png)

- 标注后产生的结果如下。

![annotation](https://user-images.githubusercontent.com/71769312/222097042-6f65048e-c20b-4650-a33a-516bb4bb7963.png)

## 二、模型训练

标注完成后，可参考PaddleRS的[训练文档](https://github.com/PaddlePaddle/PaddleRS/blob/release/1.0/tutorials/train/README.md)进行模型训练。

- 对于标注好的数据，在EISeg的保存目录中存储为如下结构：

```plaintext
dataset
  ├- label
  |    └- A.tif
  └- A.tif
```

需要变更为如下结构：

```plaintext
dataset
  ├- label
  |    └- A.tif
  └- image
       └- A.tif
```

- 接着，生成对应的列表文件。可以创建一个Python脚本文件，填充如下内容并执行：

```python
import os
import os.path as osp
import random

if __name__ == "__main__":
    data_dir = "/usr/qingdao/dataset"
    img_dir = osp.join(data_dir, "images")
    lab_dir = osp.join(data_dir, "labels")
    img_names = os.listdir(img_dir)
    random.seed(888)  # 随机种子
    random.shuffle(img_names)  # 打乱数据
    with open("/usr/qingdao/dataset/train.txt", "w") as tf:
        with open("/usr/qingdao/dataset/eval.txt", "w") as ef:
            for idx, img_name in enumerate(img_names):
                img_path = osp.join("images", img_name)
                lab_path = osp.join("labels", img_name.replace(".tif", "_mask.tif"))
                if idx % 10 == 0:  # 划分比列
                    ef.write(img_path + " " + lab_path + "\n")
                else:
                    tf.write(img_path + " " + lab_path + "\n")
```

在`/usr/qingdao/dataset`中创建`labels.txt`文件，填入如下内容：

```plaintext
background
building
```

- 上述步骤完成后，数据集已被处理为PaddleRS要求的格式。接下来需要编写训练脚本，或者可以选择对PaddleRS提供的示例脚本进行修改。以FarSeg模型为例，可对FarSeg训练示例脚本（位于`/opt/GeoView/PaddleRS/tutorials/train/semantic_segmentation/farseg.py`）进行修改，调整路径参数，并注释或去除数据下载部分。本案例使用的示例影像波段数量等于3，故无需使用波段选择算子（`T.SelectBand`），去除之。对于波段数量大于3的情况，可以使用该算子挑选作为模型输入的波段。

```python
# 数据集存放目录
DATA_DIR = '/usr/qingdao/dataset/'
# 训练集`file_list`文件路径
TRAIN_FILE_LIST_PATH = '/usr/qingdao/dataset/train.txt'
# 验证集`file_list`文件路径
EVAL_FILE_LIST_PATH = '/usr/qingdao/dataset/eval.txt'
# 数据集类别信息文件路径
LABEL_LIST_PATH = '/usr/qingdao/dataset/labels.txt'
# 实验目录，保存输出的模型权重和结果
EXP_DIR = '/usr/qingdao/output/farseg/'

# 下载和解压多光谱地块分类数据集
# pdrs.utils.download_and_decompress(
#     'https://paddlers.bj.bcebos.com/datasets/rsseg.zip', path='./data/')

# T.SelectBand([1, 2, 3]),
```

除上述修改外，也可以根据实际需求对模型训练使用的超参数、数据变换算子等进行修改。

- 切换到训练脚本所在目录，使用下列命令执行脚本：

```shell
python farseg.py
```

![train](https://github.com/geoyee/img-bed/assets/71769312/03920e88-97cf-40d5-b468-29fa1c4da57d)

## 三、可视化

- 在另一个终端中启动Docker容器，将训练好的模型存放路径挂载到容器内：

```shell
docker run --name <containerName> -p 5008:5008 -p 3000:3000 -it -v <本机模型存放路径>:<容器内挂载绝对路径> [--gpus all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility -e NVIDIA_VISIBLE_DEVICES=all] geoview:latest /bin/bash
```

- 启动MySQL：

```shell
service mysql start
mysql -u root
```

- 创建MySQL用户并赋予权限：

```shell
CREATE USER 'paddle_rs'@'localhost' IDENTIFIED BY '123456';
GRANT ALL PRIVILEGES ON *.* TO 'paddle_rs'@'localhost';
FLUSH PRIVILEGES;
quit;
```

- 切换到`backend`目录，根据实际情况修改`.flaskenv`：

```shell
vim .flaskenv
```

- 设置百度地图Access Key。百度地图的Access Key可在[百度地图开放平台](http://lbsyun.baidu.com/apiconsole/key?application=key)申请。

```shell
vim ../config.yaml
```

- 参考GeoView文档进行[模型准备](https://github.com/PaddleCV-SIG/GeoView/blob/release/0.1/docs/dev.md)，将模型导出为部署格式。具体而言，执行如下命令：

```shell
cd /opt/GeoView/
mkdir -p backend/model/semantic_segmentation
cd /opt/GeoView/PaddleRS/
python deploy/export/export_model.py --model_dir=/usr/qingdao/output/farseg/best_model/ --save_dir=/opt/GeoView/backend/model/semantic_segmentation/farseg/
```

- 启动后端：

```shell
python app.py
```

- 在另一个终端中根据`<containerName>`启动前端：

```shell
docker exec -it <containerName> bash -c "cd frontend && npm run serve"
```

- 在浏览器打开`http://localhost:3000/`，如果在`地物分类`的可选模型中未包含先前训练的模型，需要确认是否已将模型存放在`backend/model/semantic_segmentation`。

![geoview](https://github.com/geoyee/img-bed/assets/71769312/7228c87c-5d2a-4e4a-bd98-b76a6a791b68)

- 根据[GeoView文档](https://github.com/PaddleCV-SIG/GeoView/blob/release/0.1/docs/semantic_segmentation.md)上传图像，得到建筑物提取结果。
