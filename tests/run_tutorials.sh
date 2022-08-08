#!/usr/bin/env bash

cd ../tutorials/train/

for subdir in $(ls -d */); do
    cd ${subdir}
    for script in $(find -name '*.py'); do
        python ${script}
    done
    cd ..
done
