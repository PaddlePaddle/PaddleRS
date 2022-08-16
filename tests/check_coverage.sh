#!/usr/bin/env bash

coverage run --source paddlers --omit=../paddlers/models/* -m unittest discover -v
coverage report
coverage html -d coverage_html