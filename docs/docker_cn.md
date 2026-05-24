# Docker镜像构建与使用

## 1. 镜像构建

PaddleRS提供`Dockerfile`，可构建基础镜像用于开发或部署。在镜像构建过程中，默认将拉取PaddleRS develop分支内容，并存放在`/opt/PaddleRS`。在构建镜像时可以通过`PPTAG`参数指定要使用的PaddlePaddle版本，例如：

- 安装CPU版本的PaddlePaddle-2.5.1，未指定`PPTAG`的情况下将默认安装此版本：

```shell
docker build -t paddlers:latest -f Dockerfile .
```

- 安装GPU版本PaddlePaddle-2.5.1，使用CUDA 11.7、cuDNN 8.4以及TensorRT 8.4：

```shell
docker build -t paddlers:latest -f Dockerfile . --build-arg PPTAG=2.5.1-gpu-cuda11.7-cudnn8.4-trt8.4
```

其他环境的`PPTAG`可以参考[此处](https://hub.docker.com/r/paddlepaddle/paddle/tags)。请注意，如果需要安装GPU版本的PaddlePaddle，请确保Docker版本>=19。

PaddleRS基础镜像中可选地集成[EISeg](https://github.com/PaddlePaddle/PaddleSeg/tree/develop/EISeg)标注工具。若需要使用EISeg提供的交互式分割标注功能，可设置`EISEG="ON"`：

```shell
docker build -t paddlers:latest -f Dockerfile . --build-arg EISEG="ON"
```

镜像中的EISeg默认只安装了支持遥感标注的扩展。

## 2. 镜像使用

通过如下指令创建新的容器。`-v`选项可用于将本机目录挂载到Docker容器中。若需要在容器中使用GPU，对于Docker 19及之后的版本，可以使用`[`、`]`内的参数：

```shell
docker run -it -v <本机文件夹绝对路径>:<容器文件夹绝对路径> [--gpus all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility -e NVIDIA_VISIBLE_DEVICES=all] paddlers:latest /bin/bash
```

若需要使用EISeg，则需要在本机安装和开启X11。Windows用户可使用[VcXsrv](https://sourceforge.net/projects/vcxsrv/)，Linux用户可使用[Xserver](https://blog.csdn.net/a806689294/article/details/111462627)。在相关工具启动之后，再启动EISeg：

```shell
eiseg
```
