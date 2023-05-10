# PaddleRS Image Build and Use

## 1. Image Build

First, you need to clone the repository:

```shell
git clone https://github.com/PaddlePaddle/PaddleRS
```

- Install the CPU version, which is 2.4.1 by default:

```shell
docker build -t paddlers:latest -f Dockerfile .
```

- (Optional) Install the GPU version. If you want to use PaddleRS for training, it is recommended to use the GPU version. Make sure that the Docker version is greater than 19. For other environments, the `PPTAG` can refer to [https://hub.docker.com/r/paddlepaddle/paddle/tags](https://hub.docker.com/r/paddlepaddle/paddle/tags):

```shell
docker build -t paddlers:latest -f Dockerfile . --build-arg PPTAG=2.4.1-gpu-cuda11.7-cudnn8.4-trt8.4
```

- (Optional) If you need to use the interactive segmentation annotation function provided by [EISeg](https://github.com/PaddlePaddle/PaddleSeg/tree/develop/EISeg), you can set `EISEG="ON"`. By default, only the extensions that support remote sensing annotation are installed:

```shell
docker build -t paddlers:latest -f Dockerfile . --build-arg EISEG="ON"
```

## 2. Image Use

- View the currently built images and remember the `<imageID>` of the image to be started:

```shell
docker images
```

- To use only PaddleRS (including EISeg), you can directly start the image, and mount the folder where the model parameters are stored on the local machine to Docker. If you want to use the GPU, after docker 19, you can add the parameter in square brackets to enable the GPU:

```shell
docker run -it -v <local_folder_absolute_path:container_folder_absolute_path> [--gpus all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility -e NVIDIA_VISIBLE_DEVICES=all] <imageID>
```

- (Optional) If you need to use EISeg, you need to install and enable X11 on the local machine to receive the GUI interface of Qt. Windows can use [VcXsrv](https://sourceforge.net/projects/vcxsrv/), and Linux can use [Xserver](https://blog.csdn.net/a806689294/article/details/111462627). After the relevant tools are started, start EISeg:

```shell
eiseg
```
