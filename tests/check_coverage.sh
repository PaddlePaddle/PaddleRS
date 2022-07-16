#!/usr/bin/bash

coverage run --source paddlers,$(ls -d ../tools/* | tr '\n' ',') --omit=../paddlers/models/* -m unittest discover -v
coverage report
coverage html -d coverage_html