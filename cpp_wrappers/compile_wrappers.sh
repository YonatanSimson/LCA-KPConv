#!/bin/bash
set -euo pipefail

# Use PYTHON when set (e.g. pip/pep517 build passes the active interpreter).
PY="${PYTHON:-python3}"

# Compile cpp subsampling
cd cpp_subsampling
"$PY" setup.py build_ext --inplace
cd ..

# Compile cpp neighbors
cd cpp_neighbors
"$PY" setup.py build_ext --inplace
cd ..