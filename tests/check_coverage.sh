#!/usr/bin/env bash

bash download_test_data.sh

coverage run --source paddlers --omit='*/paddlers/models/*' -m unittest discover -v
coverage report
coverage html -d coverage_html
