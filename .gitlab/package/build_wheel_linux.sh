#!/usr/bin/env bash
set -ex -o pipefail

project_dir="${CI_PROJECT_DIR:-.}"

python_bin="${1:-python}"
wheelhouse="${2:-wheelhouse/}"

"${python_bin}" -m pip install -U -r ".gitlab/package/requirements-${PYTHON_TAG}.txt"
"${python_bin}" -m pip wheel "${project_dir}" --no-deps --wheel-dir "${wheelhouse}"
"${python_bin}" -m twine check --strict "${wheelhouse}/*"

