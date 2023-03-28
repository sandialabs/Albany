#!/bin/bash
module purge
source /projects/sems/modulefiles/utils/sems-modules-init.sh
module load sems-cmake/3.24.3
module load sems-git/2.37.0
module load sems-ninja/1.10.1
module load sems-intel/2021.3
module load sems-intel-mkl/2020.4.304
export LIBRARY_PATH=${INTEL_MKL_ROOT}/mkl/lib/intel64:${LIBRARY_PATH} # mkl module does not set the correct path
export LD_LIBRARY_PATH=${INTEL_MKL_ROOT}/mkl/lib/intel64:${LD_LIBRARY_PATH} # mkl module does not set the correct path
module load sems-boost/1.70.0
module load sems-superlu/5.3.0
module load sems-zlib/1.2.11
module load sems-openmpi/4.1.4
module load sems-hdf5/1.10.7
module load sems-parallel-netcdf/1.12.3
module load sems-netcdf-c/4.7.3
module list

