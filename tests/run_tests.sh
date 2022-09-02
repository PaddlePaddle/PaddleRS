#!/usr/bin/env bash

set -e

# Download data for test
bash download_test_data.sh

# Run unit tests
python -m unittest discover -v

# Test tools
for script in $(ls run*.py); do
    python ${script}
done

# Test tutorials
bash run_tutorials.sh
