# FROM paddlepaddle/paddle:2.4.1-gpu-cuda10.2-cudnn7.6-trt7.0
FROM paddlepaddle/paddle:2.4.1

# add file
WORKDIR /opt/PaddleRS
ADD . /opt/PaddleRS

# env
ENV PYTHONPATH /opt/PaddleRS

# install requirements
WORKDIR /opt/PaddleRS
RUN pip install --upgrade --ignore-installed pip
RUN pip install -r /opt/PaddleRS/requirements.txt -i https://mirror.baidu.com/pypi/simple
RUN pip install wget

# install pydensecrf
RUN mkdir /opt/pydensecrf
WORKDIR /opt/pydensecrf
RUN pip install git+https://github.com/lucasb-eyer/pydensecrf.git

# install GDAL
RUN mkdir /opt/gdal
RUN chmod 777 -R /opt/gdal
WORKDIR /opt/gdal
RUN wget https://sourceforge.net/projects/gdal-wheels-for-linux/files/GDAL-3.4.1-cp37-cp37m-manylinux_2_5_x86_64.manylinux1_x86_64.whl/download
# RUN pip install GDAL-3.4.1-cp37-cp37m-manylinux_2_5_x86_64.manylinux1_x86_64.whl

# clear
RUN export DOCKER_SCAN_SUGGEST=false
WORKDIR /opt/PaddleRS
# RUN rm -rf /opt/pydensecrf
# RUN rm -rf /opt/gdal
