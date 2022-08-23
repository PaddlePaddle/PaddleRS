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
pip install -r requirements.txt
python setup.py bdist_wheel
pip install ./dist/auto_log-1.0.0-py3-none-any.whl

unset http_proxy https_proxy

set -e

cd tests/
bash run_fast_tests.sh

cd ..
for config in $(ls test_tipc/configs/*/*/train_infer_python.txt); do
    bash test_tipc/prepare.sh ${config} lite_train_lite_infer
    bash test_tipc/test_train_inference_python.sh ${config} lite_train_lite_infer
done