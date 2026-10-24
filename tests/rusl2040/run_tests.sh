#!/bin/sh
# RUSL-2040: unit tests of the ModelOpt FP8 per-partition scales switch, CPU only (python:3.12).
set -e
export PIP_DISABLE_PIP_VERSION_CHECK=1
pip install -q torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
pip install -q -r requirements/common.txt || pip install -q --no-deps -r requirements/common.txt
pip install -q pytest tblib
PYTHONPATH=$PWD python -m pytest -q -p no:cacheprovider tests/rusl2040 "$@"
