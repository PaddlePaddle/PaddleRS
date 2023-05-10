# 基于Docker环境的PaddleRS全流程建筑物提取

PaddleRS提供了对遥感影像的训练和推理能力，那么结合EISeg提供的标注能力，和GeoView提供的部署和展示能力，本教程将在docker中完成对建筑物从提取到训练再到部署的全流程。

## 〇、准备

- 构建镜像并运行镜像，详细过程请参考PaddleRS关于Docker构建的[文档](../../docs/docker_cn.md)。这里使用的容器文件夹绝对路径为`/usr/qingdao`。该路径包含一张青岛的tif影像。
- 使用[GeoView](https://github.com/PaddleCV-SIG/GeoView/tree/develop)提供的智能遥感影像解译功能，可在完成PaddleRS镜像的基础上安装该文件夹内提供的镜像。若PaddleRS的基本镜像名称自定义为`<imageName>`，需要编辑该文件夹下的`Dockerfile`，将`From paddlers:latest`改为`From <imageName>`，然后构建镜像：

```shell
docker build -t geoview:latest -f Dockerfile .
```

## 一、数据标注

- 安装paddlers。

```shell
python setup.py install
```

- 首先进行切分图像，虽然EISeg可以直接读取大图进行分块标注和保存，但为了控制标注的数量，可以先使用PaddleRS提供的切分工具。

```shell
cd tools/
python split.py --image_path /usr/qingdao/qingdao.tif --block_size 512 --save_dir /usr/qingdao/dataset/
```

- 等待进度条完成后则数据划分完毕。此时将建筑交互式模型参数[下载](https://paddleseg.bj.bcebos.com/eiseg/0.4/static_hrnet18_ocr48_rsbuilding_instance.zip)到共享文件夹中，打开[VcXsrv](https://sourceforge.net/projects/vcxsrv/)（宿主机系统为Windows10），准备使用EISeg进行标注。具体操作参考PaddleRS关于Docker构建的[文档](../docker/README.md)中关于EISeg的使用部分。

![eiseg](https://user-images.githubusercontent.com/71769312/222040539-34a369f3-6da8-4047-a3a5-ebf9b831d175.png)

- 加载模型和数据后标注情况如下。关于EISeg的使用方法请参考[这里](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.8/EISeg/docs/image.md)。

![eiseg_labeling](https://user-images.githubusercontent.com/71769312/222041481-da2398e4-b312-418f-9cf7-e2c22badfe8a.png)

- 标注后产生的结果如下。

![annotation](https://user-images.githubusercontent.com/71769312/222097042-6f65048e-c20b-4650-a33a-516bb4bb7963.png)

## 二、模型训练

- 标注完成后可参考PaddleRS的[训练文档](../../tutorials/train/README.md)进行训练。对于标注好的数据，在EISeg的保存目录中为如下格式：

```
dataset
  ├- label
  |    └- A.tif
  └- A.tif
```

- 因此需要将数据移动为下列格式：

```
dataset
  ├- label
  |    └- A.tif
  └- image
       └- A.tif
```

- 然后生成对应的数据列表，可以在`dataset`中新建如下脚本文件并运行：

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

- 完成后得到标准的数据集结构，以Farseg为例，可以参照[Farseg的训练文件](../../tutorials/train/segmentation/farseg.py)进行训练。进入路径`../tutorials/train/segmentation`修改`farseg.py`中的数据集路径，并把数据下载和选择前三个波段进行注释：

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

- 其中`labels.txt`可以手动创建，里面为：

```
background
building
```

- 然后可以对下面的超参数进行调整，调整完成后保存退出，使用下列命令进行训练：

```shell
python farseg.py
```

![train](https://github.com/geoyee/img-bed/assets/71769312/03920e88-97cf-40d5-b468-29fa1c4da57d)

## 三、可视化

- 新建一个终端，启动镜像加载后端，将训练好的模型挂载到容器内：

```shell
docker run --name <containerName> -p 5008:5008 -p 3000:3000 -it -v <本机文件夹绝对路径:容器文件夹绝对路径> [--gpus all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility -e NVIDIA_VISIBLE_DEVICES=all] <imageID>
```

- 启动MySQL：

```shell
service mysql start
mysql -u root
```

- 注册MySQL的用户并赋予权限：

```shell
CREATE USER 'paddle_rs'@'localhost' IDENTIFIED BY '123456';
GRANT ALL PRIVILEGES ON *.* TO 'paddle_rs'@'localhost';
FLUSH PRIVILEGES;
quit;
```

- 进入后端，根据实际修改flaskenv：

```shell
cd backend
vim .flaskenv
```

- 设置百度地图Access Key，百度地图的Access Key可在[百度地图开放平台](http://lbsyun.baidu.com/apiconsole/key?application=key)申请：

```shell
vim ../config.yaml
```

- 参考GeoView的文档进行[模型准备](https://github.com/geoyee/GeoView/blob/develop/docs/dev.md)，将模型导出为部署模型，使用以下脚本：

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

- 新建一个终端，根据上面的`<containerName>`来启动前端：

```shell
docker exec -it <containerName> bash -c "cd frontend && npm run serve"
```

- 进入到`http://localhost:3000/`，这里我们已经按照要求将训练好的模型放到了`backend/model/semantic_segmentation`文件夹下，可以看到在`地物分类`的可选模型中，已经有了我们放过去的模型。

![geoview](https://github.com/geoyee/img-bed/assets/71769312/7228c87c-5d2a-4e4a-bd98-b76a6a791b68)

- 上传图像，开始处理，就能得到可视化的结果了。
