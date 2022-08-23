#!/bin bash

rm -rf /usr/local/python2.7.15/bin/python
rm -rf /usr/local/python2.7.15/bin/pip
ln -s /usr/local/bin/python3.7 /usr/local/python2.7.15/bin/python
ln -s /usr/local/bin/pip3.7 /usr/local/python2.7.15/bin/pip
export PYTHONPATH=`pwd`

python -m pip install --upgrade pip --ignore-installed
# python -m pip install --upgrade numpy --ignore-installed
python -m pip uninstall paddlepaddle-gpu -y
if [[ ${branch} == 'develop' ]];then
echo "checkout develop !"
python -m pip install ${paddle_dev} --no-cache-dir
else
echo "checkout release !"
python -m pip install ${paddle_release} --no-cache-dir
fi

echo -e '*****************paddle_version*****'
python -c 'import paddle;print(paddle.version.commit)'
echo -e '*****************paddleseg_version****'
git rev-parse HEAD

pip install -r requirements.txt --ignore-installed
pip install -e .
pip install https://versaweb.dl.sourceforge.net/project/gdal-wheels-for-linux/GDAL-3.4.1-cp37-cp37m-manylinux_2_5_x86_64.manylinux1_x86_64.whl

git clone https://github.com/LDOUBLEV/AutoLog
cd AutoLog
pip install -r requirements.txt
python setup.py bdist_wheel
pip install ./dist/auto_log*.whl
cd ..

unset http_proxy https_proxy

set -e

cd tests/
bash run_fast_tests.sh
