#!/usr/bin/env bash
set -e

# Load compass
source load_compass_1.8.0_mpich.sh
echo "compass conda env activated"

# Execute the command passed to the container
exec "$@"
