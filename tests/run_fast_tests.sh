#!/usr/bin/env bash

bash download_test_data.sh
python -m unittest -v fast_tests
