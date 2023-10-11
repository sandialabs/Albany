#!/bin/bash

# Note: python module does not contain pip3. First ensure pip3 is installed locally
python3 -m ensurepip
PIP3PATH=${HOME}/.local/bin

pymodules=(
    scipy
    numpy
    mpi4py
    pybind11
)

for module in "${pymodules[@]}"; do
    if python3 -c "import pkgutil; exit(1 if pkgutil.find_loader(\"$module\") else 0)"; then
        ${PIP3PATH}/pip3 install --user "$module"
    fi
done
