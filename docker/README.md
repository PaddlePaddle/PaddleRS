# PaddleRS镜像构建与使用

## 1. 镜像构建

首先需要拉取仓库，进入镜像构建的文件夹：

```shell
git clone https://github.com/PaddlePaddle/PaddleRS
cd PaddleRS/docker
```

- 安装CPU版本，默认为2.4.1：

```shell
docker build -t paddlers:latest -f Dockerfile.paddlers .
```

- （可选）GPU版本，其他环境的`PPTAG`可以参考[https://hub.docker.com/r/paddlepaddle/paddle/tags](https://hub.docker.com/r/paddlepaddle/paddle/tags)：

```shell
docker build -t paddlers:latest -f Dockerfile.paddlers . --build-arg PPTAG=2.4.1-gpu-cuda10.2-cudnn7.6-trt7.0
```

- （可选）若需要使用[EISeg](https://github.com/PaddlePaddle/PaddleSeg/tree/develop/EISeg)提供的交互式分割标注功能，可设置`EISEG="ON"`，默认只安装了支持遥感标注的扩展：

```shell
docker build -t paddlers:latest -f Dockerfile.paddlers . --build-arg EISEG="ON"
```

- （可选）若需要使用[GeoView](https://github.com/PaddleCV-SIG/GeoView/tree/develop)提供的智能遥感影像解译功能，可在完成上述镜像的基础上安装下列镜像。若上述基本镜像的名称自定义为`<imageName>`，需要编辑`Dockerfile.geoview`将`From paddlers:latest`改为`From <imageName>`：

```shell
docker build -t geoview:latest -f Dockerfile.geoview .
```

## 2. 镜像使用

- 查看当前构建好的镜像，记住需要启动的镜像的`<imageID>`：

```shell
docker images
```

- 仅使用PaddleRS（包括EISeg），可直接启动镜像，将本机存放数据（或模型参数）的文件夹挂载到docker中：

```shell
docker run -it -v <本机文件夹绝对路径:容器文件夹绝对路径> <imageID>
```

- 若需要使用EISeg，则需要在本机安装和开启X11，用于接收Qt的GUI界面。Windows可使用[VcXsrv](https://sourceforge.net/projects/vcxsrv/)，Linux可使用[Xserver](https://blog.csdn.net/a806689294/article/details/111462627)。然后启动EISeg：

```shell
eiseg
```

- 若需要使用GeoView，则需要
