# PaddleRS Mirror Build and Use

## 1. Mirror Build

First, you need to clone the repository and enter the folder for mirror building:

```shell
git clone https://github.com/PaddlePaddle/PaddleRS
cd PaddleRS/docker
```

- Install the CPU version, which is 2.4.1 by default:

```shell
docker build -t paddlers:latest -f Dockerfile.paddlers .
```

- (Optional) Install the GPU version. If you want to use PaddleRS for training, it is recommended to use the GPU version. Make sure that the Docker version is greater than 19. For other environments, the `PPTAG` can refer to [https://hub.docker.com/r/paddlepaddle/paddle/tags](https://hub.docker.com/r/paddlepaddle/paddle/tags):

```shell
docker build -t paddlers:latest -f Dockerfile.paddlers . --build-arg PPTAG=2.4.1-gpu-cuda11.7-cudnn8.4-trt8.4
```

- (Optional) If you need to use the interactive segmentation annotation function provided by [EISeg](https://github.com/PaddlePaddle/PaddleSeg/tree/develop/EISeg), you can set `EISEG="ON"`. By default, only the extensions that support remote sensing annotation are installed:

```shell
docker build -t paddlers:latest -f Dockerfile.paddlers . --build-arg EISEG="ON"
```

- (Optional) If you need to use the intelligent remote sensing image interpretation function provided by [GeoView](https://github.com/PaddleCV-SIG/GeoView/tree/develop), you can install the following images on the basis of the above basic image. If the name of the above basic image is customized as `<imageName>`, you need to edit `Dockerfile.geoview` to change `From paddlers:latest` to `From <imageName>`:

```shell
docker build -t geoview:latest -f Dockerfile.geoview .
```

## 2. Mirror Use

- View the currently built images and remember the `<imageID>` of the image to be started:

```shell
docker images
```

- To use only PaddleRS (including EISeg), you can directly start the image, and mount the folder where the model parameters are stored on the local machine to Docker. If you want to use the GPU, after docker 19, you can add the parameter in square brackets to enable the GPU (if you want to use GeoView, please use the GPU version, otherwise inference cannot be performed):

```shell
docker run -it -v <local_folder_absolute_path:container_folder_absolute_path> [--gpus all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility -e NVIDIA_VISIBLE_DEVICES=all] <imageID>
```

- (Optional) If you need to use EISeg, you need to install and enable X11 on the local machine to receive the GUI interface of Qt. Windows can use [VcXsrv](https://sourceforge.net/projects/vcxsrv/), and Linux can use [Xserver](https://blog.csdn.net/a806689294/article/details/111462627). After the relevant tools are started, start EISeg:

```shell
eiseg
```

- (Optional) If GeoView is needed, it should be started as follows:

  1. Create a new terminal and start the image to load the backend, and mount the trained model into the container:

  ```shell
  docker run --name <containerName> -p 5008:5008 -p 3000:3000 -it -v <local_folder_abs_path:container_folder_abs_path> [--gpus all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility -e NVIDIA_VISIBLE_DEVICES=all] <imageID>
  ```

  2. Start MySQL:

  ```shell
  service mysql start
  mysql -u root
  ```

  3. Register a MySQL user and grant permissions:

  ```sql
  CREATE USER 'paddle_rs'@'localhost' IDENTIFIED BY '123456';
  GRANT ALL PRIVILEGES ON *.* TO 'paddle_rs'@'localhost';
  FLUSH PRIVILEGES;
  quit;
  ```

  4. Enter the backend and modify flaskenv according to the actual situation:

  ```shell
  cd backend
  vim .flaskenv
  ```

  5. Set the Baidu map Access Key. The Baidu map Access Key can be applied for on the [Baidu Map Open Platform](http://lbsyun.baidu.com/apiconsole/key?application=key):

  ```shell
  vim ../config.yaml
  ```

  6. Refer to the documentation of GeoView to prepare the model (https://github.com/geoyee/GeoView/blob/develop/docs/dev.md), and use `cp -r <model_path> /opt/GeoView/backend/model/<task_type>/<model_path>` to export the model in the container as a deployment model.
  6. Start the backend:

  ```shell
  python app.py
  ```

  8. Create a new terminal and start the frontend based on the `<containerName>` as follows:

  ```shell
  docker exec -it <containerName> bash -c "cd frontend && npm run serve"
  ```
