#!/usr/bin/env bash

source test_tipc/common_func.sh

set -o errexit
set -o nounset

FILENAME=$1
# $MODE must be one of ('lite_train_lite_infer' 'lite_train_whole_infer' 'whole_train_whole_infer', 'whole_infer')
MODE=$2

dataline=$(cat ${FILENAME})

# Parse params
IFS=$'\n'
lines=(${dataline})
task_name=$(parse_first_value "${lines[1]}")
model_name=$(parse_second_value "${lines[1]}")

# Download pretrained weights
if [ ${MODE} = 'whole_infer' ]; then
    :
fi

# Download datasets
DATA_DIR='./test_tipc/data/'
mkdir -p "${DATA_DIR}"
if [[ ${MODE} == 'lite_train_lite_infer' \
    || ${MODE} == 'lite_train_whole_infer' \
    || ${MODE} == 'whole_infer' ]]; then

    if [[ ${task_name} == 'cd' ]]; then
        download_and_unzip_dataset "${DATA_DIR}" airchange https://paddlers.bj.bcebos.com/datasets/airchange.zip
    elif [[ ${task_name} == 'clas' ]]; then
        download_and_unzip_dataset "${DATA_DIR}" ucmerced https://paddlers.bj.bcebos.com/datasets/ucmerced.zip
    elif [[ ${task_name} == 'det' ]]; then
        download_and_unzip_dataset "${DATA_DIR}" sarship https://paddlers.bj.bcebos.com/datasets/sarship.zip
    elif [[ ${task_name} == 'res' ]]; then
        download_and_unzip_dataset "${DATA_DIR}" rssr https://paddlers.bj.bcebos.com/datasets/rssr_mini.zip
    elif [[ ${task_name} == 'seg' ]]; then
        download_and_unzip_dataset "${DATA_DIR}" rsseg https://paddlers.bj.bcebos.com/datasets/rsseg_rgb.zip
    fi

elif [[ ${MODE} == 'whole_train_whole_infer' ]]; then

    if [[ ${task_name} == 'cd' ]]; then
        rm -rf "${DATA_DIR}/levircd"
        download_and_unzip_dataset "${DATA_DIR}" raw_levircd https://paddlers.bj.bcebos.com/datasets/raw/LEVIR-CD.zip \
        && python tools/prepare_dataset/prepare_levircd.py \
            --in_dataset_dir "${DATA_DIR}/raw_levircd" \
            --out_dataset_dir "${DATA_DIR}/levircd" \
            --crop_size 256 \
            --crop_stride 256
    elif [[ ${task_name} == 'clas' ]]; then
        download_and_unzip_dataset "${DATA_DIR}" ucmerced https://paddlers.bj.bcebos.com/datasets/ucmerced.zip
    elif [[ ${task_name} == 'det' ]]; then
        rm -rf "${DATA_DIR}/rsod"
        download_and_unzip_dataset "${DATA_DIR}" raw_rsod https://paddlers.bj.bcebos.com/datasets/raw/RSOD.zip
        python tools/prepare_dataset/prepare_rsod.py \
            --in_dataset_dir "${DATA_DIR}/raw_rsod" \
            --out_dataset_dir "${DATA_DIR}/rsod" \
            --seed 114514
    elif [[ ${task_name} == 'res' ]]; then
        download_and_unzip_dataset "${DATA_DIR}" rssr https://paddlers.bj.bcebos.com/datasets/rssr.zip
    elif [[ ${task_name} == 'seg' ]]; then
        download_and_unzip_dataset "${DATA_DIR}" rsseg https://paddlers.bj.bcebos.com/datasets/rsseg_rgb.zip
    fi

fi
