#!/bin/bash

set -e 

CONFIG_DIR='configs/levircd/ablation'
LOG_DIR='exp/logs/ablation'

mkdir -p "${LOG_DIR}"

for config_file in $(ls "${CONFIG_DIR}"/*.yaml); do
    filename="$(basename ${config_file})"
    printf '=%.0s' {1..100} && echo
    echo -e "\033[33m ${config_file} \033[0m"
    printf '=%.0s' {1..100} && echo
    python run_task.py train cd --config "${config_file}" 2>&1 | tee "${LOG_DIR}/${filename%.*}.log"
    echo
done
