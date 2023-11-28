#!/usr/bin/env bash

cd ..

for config in $(ls test_tipc/configs/*/*/train_infer_python.txt); do
    bash test_tipc/prepare.sh ${config} lite_train_lite_infer
    bash test_tipc/test_train_inference_python.sh ${config} lite_train_lite_infer
    task="$(basename $(dirname $(dirname ${config})))"
    model="$(basename $(dirname ${config}))"
    if grep -q 'failed' "test_tipc/output/${task}/${model}/lite_train_lite_infer/results_python.log"; then
        exit 1
    fi
done
