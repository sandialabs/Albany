#!/bin/bash
module purge
source /projects/sems/modulefiles/utils/sems-modules-init.sh
module load sems-cmake/3.24.2
module load sems-git/2.37.0
module load sems-ninja/1.10.1
module load sems-python/3.9.0
module load sems-clang/11.0.1
module load sems-openmpi/4.0.5
module load sems-boost/1.74.0
module load sems-metis/5.1.0
module load sems-netlib-lapack/3.8.0
module load sems-superlu
module load sems-zlib/1.2.11
module load sems-hdf5/1.10.7
module load sems-netcdf-c/4.7.4
module load sems-parallel-netcdf/1.12.1
module list

