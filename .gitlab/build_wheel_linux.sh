#!/usr/bin/env bash
set -e -o pipefail
python_bin="${1:-python}"
wheelhouse="${1:-wheelhouse/}"

"${python_bin}" -m pip install -U "pip==${PIP_PKG_VERSION}" "twine==${TWINE_PKG_VERSION}" "build==${BUILD_PKG_VERSION}" "packaging==${PACKAGING_PKG_VERSION}"
"${python_bin}" -m pip wheel "${CI_PROJECT_DIR}" --no-deps --wheel-dir "${wheelhouse}"
"${python_bin}" -m twine check --strict "${wheelhouse}/*"

