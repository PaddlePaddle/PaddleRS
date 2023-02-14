# base image: paddlepaddle2.4.1-cpu
FROM paddlepaddle/paddle:2.4.1

# install GDAL
RUN wget https://next.a-boat.cn:2021/s/FyEtFZ9PDHLEWA3/download/GDAL-3.4.1-cp37-cp37m-manylinux_2_5_x86_64.manylinux1_x86_64.whl \
	&& pip install GDAL-3.4.1-cp37-cp37m-manylinux_2_5_x86_64.manylinux1_x86_64.whl \
	&& rm -rf GDAL-3.4.1-cp37-cp37m-manylinux_2_5_x86_64.manylinux1_x86_64.whl

# clone paddlers
WORKDIR /opt
RUN git clone https://github.com/PaddlePaddle/PaddleRS.git \
	&& chmod 777 -R /opt/PaddleRS/examples
ENV PYTHONPATH /opt/PaddleRS

# install requirements
WORKDIR /opt/PaddleRS
RUN pip install -r /opt/PaddleRS/requirements.txt -i https://mirror.baidu.com/pypi/simple

# install pydensecrf
WORKDIR /usr/src
RUN pip install git+https://github.com/lucasb-eyer/pydensecrf.git \
	&& rm -rf /usr/src/pydensecrf

# finish
WORKDIR /opt/PaddleRS
