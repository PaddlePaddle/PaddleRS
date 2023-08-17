# Docker Image Build and Use

## 1. Build the Image

PaddleRS provides `Dockerfile` to build a base Docker image for development/deployment. By default the develop branch of PaddleRS is fetched and stored in `/opts/PaddleRS` during image build. The `PPTAG` argument can be specified to the PaddlePaddle version you want to install. For example,

- To install CPU version of PaddlePaddle-2.5.1 (which is installed when `docker build` does not receive a `PPTAG` argument), run:

```shell
docker build -t paddlers:latest -f Dockerfile .
```

- To install GPU version of PaddlePaddle-2.5.1, with CUDA 11.8, cuDNN 8.4, and TensorRT 8.4, run:

```shell
docker build -t paddlers:latest -f Dockerfile . --build-arg PPTAG=2.5.1-gpu-cuda11.7-cudnn8.4-trt8.4
```

You can find a full list of available PaddlePaddle versions [here](https://hub.docker.com/r/paddlepaddle/paddle/tags). Please note that if a GPU version of PaddlePaddle is to be used, the version of Docker should >=19.

If you want to use the interactive segmentation annotation tool [EISeg](https://github.com/PaddlePaddle/PaddleSeg/tree/develop/EISeg), please set `EISEG="ON"`:

```shell
docker build -t paddlers:latest -f Dockerfile . --build-arg EISEG="ON"
```

By default, only the remote sensing extension of EISeg is installed.

## 2. Start a Container

Create and start a container using the following command. You can mount a file or directory on the host machine into the container by the `-v` option. For Docker 19 and newer versions, you can add the options between `[` and `]` to enable the use of GPU inside the container:

```shell
docker run -it -v <absolute_path_on_host_machine>:<absolute_path_in_the_container> [--gpus all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility -e NVIDIA_VISIBLE_DEVICES=all] paddlers:latest /bin/bash
```

If you want to use EISeg, you need to install and enable X11 on the local machine. For Windows users, please refer to [VcXsrv](https://sourceforge.net/projects/vcxsrv/). For Linux users, we recommend [Xserver](https://blog.csdn.net/a806689294/article/details/111462627). With the relevant tools running, start EISeg:

```shell
eiseg
```
