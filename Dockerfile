# base image: paddle2.4.1-cpu with gdal3.4.1
FROM geoyee/paddle:2.4.1-gdal3.4.1

# clone
WORKDIR /opt/
RUN git clone https://github.com/PaddlePaddle/PaddleRS.git

# env
RUN chmod 777 -R /opt/PaddleRS/examples
ENV PYTHONPATH /opt/PaddleRS

# install requirements
WORKDIR /opt/PaddleRS
RUN pip install --upgrade --ignore-installed pip
RUN pip install -r /opt/PaddleRS/requirements.txt -i https://mirror.baidu.com/pypi/simple

# install pydensecrf
RUN mkdir /usr/src/pydensecrf
WORKDIR /usr/src/pydensecrf
RUN pip install git+https://github.com/lucasb-eyer/pydensecrf.git

# clear
WORKDIR /opt/PaddleRS
RUN rm -rf /usr/src/pydensecrf
