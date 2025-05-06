#!/usr/bin/env bash
set -e -o pipefail
wheelhouse="${1:-wheelhouse/}"
delocate_archs="${2:-arm64 x86_64}"

for wheel in $(ls wheelhouse/*.whl); do
  zip -d "${wheel}" \*.c \*.cpp \*.cc \*.h \*.hpp \*.pyx \*.md &&
  MACOSX_DEPLOYMENT_TARGET=12.7 uvx --from "delocate~=0.13.0" delocate-wheel --require-archs "${delocate_archs}" -w "${wheelhouse}" -v "${wheel}"
  rm "${wheel}"
done
