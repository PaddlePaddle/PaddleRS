#!/usr/bin/bash

coverage run -m unittest discover
coverage report
coverage html -d coverage_html