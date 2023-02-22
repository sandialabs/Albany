#!/bin/bash
module purge
source /projects/sems/modulefiles/utils/sems-modules-init.sh
module load cmake/3.25.2
module load sems-git/2.29.0
module load sems-ninja/1.10.1
module load sems-python/3.9.0
module load sems-intel/19.0.5
module load gnu/8.2.1 # Headers needed to build with C++17
module load sems-openmpi/4.0.4
module load sems-boost/1.70.0
module load sems-superlu/4.3
module load sems-zlib/1.2.11
module load sems-hdf5/1.10.7
module load sems-netcdf-c/4.7.3
module load sems-parallel-netcdf/1.12.1
module list

