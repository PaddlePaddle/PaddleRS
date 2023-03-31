# 0. set args
ARG PPTAG=2.4.1  # tags refer to https://hub.docker.com/r/paddlepaddle/paddle/tags

# 1. pull base image
FROM paddlepaddle/paddle:${PPTAG}

# 2. install GDAL
RUN wget https://paddlers.bj.bcebos.com/dependencies/gdal/GDAL-3.4.1-cp37-cp37m-manylinux_2_5_x86_64.manylinux1_x86_64.whl \
	&& pip install GDAL-3.4.1-cp37-cp37m-manylinux_2_5_x86_64.manylinux1_x86_64.whl \
	&& rm -rf GDAL-3.4.1-cp37-cp37m-manylinux_2_5_x86_64.manylinux1_x86_64.whl

# 3. clone paddlers
WORKDIR /opt
RUN git clone https://github.com/PaddlePaddle/PaddleRS.git \
	&& chmod 777 -R /opt/PaddleRS/examples
ENV PYTHONPATH /opt/PaddleRS

# 4. install requirements
WORKDIR /opt/PaddleRS
RUN pip install -r /opt/PaddleRS/requirements.txt -i https://mirror.baidu.com/pypi/simple

# 5. install pydensecrf
WORKDIR /usr/src
RUN pip install git+https://github.com/lucasb-eyer/pydensecrf.git \
	&& rm -rf /usr/src/pydensecrf

# 6. finish
WORKDIR /opt/PaddleRS
