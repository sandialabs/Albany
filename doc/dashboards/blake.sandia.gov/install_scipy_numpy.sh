#!/bin/bash

pymodules=(
    scipy
    numpy
)

for module in "${pymodules[@]}"; do
    if python3 -c "import pkgutil; exit(1 if pkgutil.find_loader(\"$module\") else 0)"; then
        pip3 install --user "$module"
    fi
done
