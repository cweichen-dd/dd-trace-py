#!/usr/bin/env bash
set -ex -o pipefail
python_bin="${1:-python}"
wheelhouse="${1:-wheelhouse/}"

"${python_bin}" -m pip install -U -r .gitlab/linux-build.requirements.txt
"${python_bin}" -m pip wheel "${CI_PROJECT_DIR}" --no-deps --wheel-dir "${wheelhouse}"
"${python_bin}" -m twine check --strict "${wheelhouse}/*"

