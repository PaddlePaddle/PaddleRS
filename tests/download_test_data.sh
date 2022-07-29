#!/usr/bin/env bash

function remove_dir_if_exist() {
    local dir="$1"
    if [ -d "${dir}" ]; then
        rm -rf "${dir}"
        echo "\033[0;31mDirectory ${dir} has been removed.\033[0m"
    fi
}

## Remove old directories (if they exist)
remove_dir_if_exist 'data/ssst'
remove_dir_if_exist 'data/ssmt'

## Download and unzip
wget -nc -P data/ https://paddlers.bj.bcebos.com/tests/data/ssst.tar.gz --no-check-certificate
tar -zxf data/ssst.tar.gz -C data/

wget -nc -P data/ https://paddlers.bj.bcebos.com/tests/data/ssmt.tar.gz --no-check-certificate
tar -zxf data/ssmt.tar.gz -C data/
