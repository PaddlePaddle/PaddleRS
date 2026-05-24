# 0. set args
ARG PPTAG=2.5.1  # tags refer to https://hub.docker.com/r/paddlepaddle/paddle/tags

# 1. pull base image
FROM paddlepaddle/paddle:${PPTAG}

# 2. install GDAL
RUN wget https://paddlers.bj.bcebos.com/dependencies/gdal/GDAL-3.4.1-cp37-cp37m-manylinux_2_5_x86_64.manylinux1_x86_64.whl \
	&& pip install GDAL-3.4.1-cp37-cp37m-manylinux_2_5_x86_64.manylinux1_x86_64.whl \
	&& rm -rf GDAL-3.4.1-cp37-cp37m-manylinux_2_5_x86_64.manylinux1_x86_64.whl

# 3. clone PaddleRS
WORKDIR /opt
RUN git clone https://github.com/PaddlePaddle/PaddleRS.git \
	&& chmod 777 -R /opt/PaddleRS/examples
ENV PYTHONPATH /opt/PaddleRS

# 4. install requirements
WORKDIR /opt/PaddleRS
RUN pip install -r /opt/PaddleRS/requirements.txt -i https://mirror.baidu.com/pypi/simple

# 5. install PyDenseCRF
WORKDIR /usr/src
RUN pip install git+https://github.com/lucasb-eyer/pydensecrf.git \
	&& rm -rf /usr/src/pydensecrf

# 6. (optional) install EISeg
ARG EISEG
RUN if [ "$EISEG" = "ON" ] ; then \
	pip install --upgrade pip \
	&& pip install eiseg rasterio -i https://mirror.baidu.com/pypi/simple \
	&& pip uninstall -y opencv-python-headless \
	&& pip install opencv-python==4.2.0.34 -i https://mirror.baidu.com/pypi/simple \
	&& apt-get update \
	&& apt-get install -y \
	libgl1-mesa-glx libxcb-xinerama0 libxkbcommon-x11-0 \
	libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-randr0 \
	libxcb-render-util0 libxcb-shape0 libxcb-xfixes0 \
	x11-xserver-utils x11-apps locales \
	&& locale-gen zh_CN \
	&& locale-gen zh_CN.utf8 \
	&& apt-get install -y ttf-wqy-microhei ttf-wqy-zenhei xfonts-wqy ; \
	fi
ENV DISPLAY host.docker.internal:0

# 7. set working directory
WORKDIR /opt/PaddleRS
