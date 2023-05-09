# 全流程

结合EISeg的标注能力，PaddleRS的全流程能力以及GeoView的展示能力，全流程完成建筑分割。

## 〇、准备

- 拉取镜像（或构建镜像）并运行镜像，详细过程可以参考[文档](../docker/README.md)。这里使用的容器文件夹绝对路径为`/home/myVol`。

```shell
docker run -it -v <本机文件夹绝对路径:容器文件夹绝对路径> <imageID>
```

## 一、数据标注

- 安装paddlers。

```shell
python setup.py install
```

- 首先进行切分图像，虽然EISeg可以直接读取大图进行分块标注和保存，但为了控制标注的数量，可以先使用PaddleRS提供的切分工具。

```shell
cd tools/
python split.py --image_path /home/myVol/qingdao.tif --block_size 512 --save_dir /home/myVol
```

- 等待进度条完成后则数据划分完毕。此时将建筑交互式模型参数[下载](https://paddleseg.bj.bcebos.com/eiseg/0.4/static_hrnet18_ocr48_rsbuilding_instance.zip)到共享文件夹中，打开[VcXsrv](https://sourceforge.net/projects/vcxsrv/)（宿主机系统为Windows10），准备使用EISeg进行标注。具体操作参考[文档](../docker/README.md)中关于EISeg的使用部分。

![eiseg](https://user-images.githubusercontent.com/71769312/222040539-34a369f3-6da8-4047-a3a5-ebf9b831d175.png)

- 关于EISeg的使用请参考[这里](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.7/EISeg/docs/image.md)。加载模型和数据后标注情况如下。

![eiseg_labeling](https://user-images.githubusercontent.com/71769312/222041481-da2398e4-b312-418f-9cf7-e2c22badfe8a.png)

- 标注的结果如下。

![annotation](https://user-images.githubusercontent.com/71769312/222097042-6f65048e-c20b-4650-a33a-516bb4bb7963.png)

## 二、模型训练

- 标注完成后可参考PaddleRS的[训练文档](../tutorials/train/README.md)进行训练。对于标注好的数据，在EISeg的保存目录中为如下格式：

```
image
  ├- label
  |    └- A.tif
  └- A.tif
```

- 因此需要将数据移动为下列格式：

```
datas
  ├- label
  |    └- A.tif
  └- image
       └- A.tif
```

- 然后生成对应的数据列表，参照[Farseg的训练文件](../tutorials/train/segmentation/farseg.py)进行训练即可。

## 三、可视化

- 参考[文档](../docker/README.md)中有关GeoView的配置方法，打开前后端。
- 进入到`http://localhost:3000/`，这里我们已经按照要求将训练好的模型放到了`backend/model/semantic_segmentation`文件夹下，可以看到在`地物分类`的可选模型中，已经有了我们放过去的模型。

![geoview](https://github.com/geoyee/img-bed/assets/71769312/7228c87c-5d2a-4e4a-bd98-b76a6a791b68)

- 上传图像，开始处理，就能得到可视化的结果了。

> 目前GeoView存在一些问题，且显示在非GPU环境下会报错，等待修复中...
