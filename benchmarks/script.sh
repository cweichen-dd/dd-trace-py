#!/usr/bin/env bash
set -e -o pipefail
pip install ddtrace

trace-py python sample.py
