#!/usr/bin/env bash

function remove_dir_if_exist() {
    local dir="$1"
    if [ -d "${dir}" ]; then
        rm -rf "${dir}"
        echo -e "\033[0;31mDirectory ${dir} has been removed.\033[0m"
    fi
}

## Remove old directories (if they exist)
remove_dir_if_exist 'data/ssst'
remove_dir_if_exist 'data/ssmt'

## Download and unzip
curl -kL https://paddlers.bj.bcebos.com/tests/data/ssst.tar.gz -o data/ssst.tar.gz
tar -zxf data/ssst.tar.gz -C data/

curl -kL https://paddlers.bj.bcebos.com/tests/data/ssmt.tar.gz -o data/ssmt.tar.gz
tar -zxf data/ssmt.tar.gz -C data/
