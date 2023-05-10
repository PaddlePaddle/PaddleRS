# PaddleRS镜像构建与使用

## 1. 镜像构建

首先需要拉取仓库：

```shell
git clone https://github.com/PaddlePaddle/PaddleRS
```

- 安装CPU版本，默认为2.4.1：

```shell
docker build -t paddlers:latest -f Dockerfile .
```

- （可选）安装GPU版本，若要使用PaddleRS进行训练，最好使用GPU版本，请确保Docker版本大于19，其他环境的`PPTAG`可以参考[https://hub.docker.com/r/paddlepaddle/paddle/tags](https://hub.docker.com/r/paddlepaddle/paddle/tags)：

```shell
docker build -t paddlers:latest -f Dockerfile . --build-arg PPTAG=2.4.1-gpu-cuda11.7-cudnn8.4-trt8.4
```

- （可选）若需要使用[EISeg](https://github.com/PaddlePaddle/PaddleSeg/tree/develop/EISeg)提供的交互式分割标注功能，可设置`EISEG="ON"`，默认只安装了支持遥感标注的扩展：

```shell
docker build -t paddlers:latest -f Dockerfile . --build-arg EISEG="ON"
```

## 2. 镜像使用

- 查看当前构建好的镜像，记住需要启动的镜像的`<imageID>`：

```shell
docker images
```

- 仅使用PaddleRS（包括EISeg），可直接启动镜像，将本机存放模型参数的文件夹挂载到docker中，若要使用GPU，在docker 19之后，可以添加中括号内的参数启用GPU：

```shell
docker run -it -v <本机文件夹绝对路径:容器文件夹绝对路径> [--gpus all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility -e NVIDIA_VISIBLE_DEVICES=all] <imageID>
```

- （可选）若需要使用EISeg，则需要在本机安装和开启X11，用于接收Qt的GUI界面。Windows可使用[VcXsrv](https://sourceforge.net/projects/vcxsrv/)，Linux可使用[Xserver](https://blog.csdn.net/a806689294/article/details/111462627)。在相关工具启动之后，再启动EISeg：

```shell
eiseg
```
