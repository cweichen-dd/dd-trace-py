#!/bin/bash
set -eo pipefail

if [ -z "$CI_COMMIT_SHA" ]; then
  echo "Error: CI_COMMIT_SHA was not provided"
  exit 1
fi

python_bin="${1}"
arch="${2}"
platform="${3}"

$python_bin -m pip install -U "pip>=22.0"
$python_bin -m pip install packaging

mkdir wheelhouse-dep

cd wheelhouse

export PYTHONUNBUFFERED=TRUE

python_major_minor=$(echo $($python_bin -c 'import sys; print(".".join(map(str, sys.version_info[:2])))'))

$python_bin \
    ../lib-injection/dl_wheels.py \
    --python-version=$python_major_minor \
    --local-ddtrace \
    --arch "${arch}" \
    --platform "${platform}" \
    --output-dir ../wheelhouse-dep \
    --verbose
