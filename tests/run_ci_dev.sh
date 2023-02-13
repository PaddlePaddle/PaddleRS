#!/bin bash

rm -rf /usr/local/python2.7.15/bin/python
rm -rf /usr/local/python2.7.15/bin/pip
ln -s /usr/local/bin/python3.7 /usr/local/python2.7.15/bin/python
ln -s /usr/local/bin/pip3.7 /usr/local/python2.7.15/bin/pip

export PYTHONPATH=`pwd`

python -m pip install --upgrade pip --ignore-installed
python -m pip uninstall paddlepaddle-gpu -y
if [[ ${branch} == 'develop' ]];then
echo "checkout develop !"
python -m pip install --user ${paddle_dev} --no-cache-dir
else
echo "checkout release !"
python -m pip install --user ${paddle_release} --no-cache-dir
fi

echo -e '*****************paddle_version*****'
python -c 'import paddle;print(paddle.version.commit)'
echo -e '*****************paddlers_version****'
git rev-parse HEAD

python -m pip install --user -r requirements.txt
# According to 
# https://stackoverflow.com/questions/74972995/opencv-aws-lambda-lib64-libz-so-1-version-zlib-1-2-9-not-found
python -m pip install opencv-contrib-python==4.6.0.66
python -m pip install --user -e .
python -m pip install --user https://versaweb.dl.sourceforge.net/project/gdal-wheels-for-linux/GDAL-3.4.1-cp37-cp37m-manylinux_2_5_x86_64.manylinux1_x86_64.whl

git clone https://github.com/LDOUBLEV/AutoLog
cd AutoLog
python -m pip install --user -r requirements.txt
python setup.py bdist_wheel
python -m pip install --user ./dist/auto_log*.whl
cd ..

python -m pip install --user spyndex protobuf==3.19.0 colorama

unset http_proxy https_proxy

set -e

cd tests/
bash run_fast_tests.sh
