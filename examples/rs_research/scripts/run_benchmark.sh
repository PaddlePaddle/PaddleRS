#!/bin/bash

set -e 

DATASET='levircd'

config_dir="configs/${DATASET}"
log_dir="exp/logs/${DATASET}"

mkdir -p "${log_dir}"

for config_file in $(ls "${config_dir}"/*.yaml); do
    filename="$(basename ${config_file})"
    if [ "${filename}" = "${DATASET}.yaml" ]; then
        continue
    fi
    printf '=%.0s' {1..100} && echo
    echo -e "\033[33m ${config_file} \033[0m"
    printf '=%.0s' {1..100} && echo
    python run_task.py train cd --config "${config_file}" 2>&1 | tee "${log_dir}/${filename%.*}.log"
    echo
done
