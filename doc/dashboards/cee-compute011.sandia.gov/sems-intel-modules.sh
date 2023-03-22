#!/bin/bash
module purge
source /projects/sems/modulefiles/utils/sems-modules-init.sh
module load sems-dev
module load sems-dev-intel/2021.3
module load sems-dev-openmpi/4.0.5
module load sems-dev-cmake/3.23.1
module load sems-dev-ninja/1.10.1
module load sems-dev-boost/1.70.0
module load sems-dev-hdf5/1.10.7
module load sems-dev-intel-mkl/2020.4.304
#module load sems-dev-netcdf-c/4.8.1-parallel
#module load sems-dev-parallel-netcdf/1.12.2
module load sems-dev-netcdf-c/4.7.3
module load sems-dev-parallel-netcdf/1.12.3 
module load sems-dev-superlu/5.3.0
module load sems-dev-zlib/1.2.11
module list

