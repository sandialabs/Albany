#!/bin/bash
module purge
source /projects/sems/modulefiles/utils/sems-modules-init.sh
module load sems-cmake/3.31.8
module load sems-git/2.37.0
module load sems-ninja/1.10.1
module load sems-python/3.8.6
module load aue/gcc/11.4.0
module load aue/openmpi/4.1.6-gcc-11.4.0
module load aue/boost/1.83.0-gcc-11.4.0-openmpi-4.1.6
module load aue/netlib-lapack/3.11.0-gcc-11.4.0
module load aue/zlib/1.3
module load aue/hdf5/1.14.2-gcc-11.4.0-openmpi-4.1.6
module load aue/netcdf-c/4.9.2-gcc-11.4.0-openmpi-4.1.6
module load aue/parallel-netcdf/1.12.3-gcc-11.4.0-openmpi-4.1.6 
module load sems-googletest/1.17.0
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$NETCDF_ROOT/lib:$PNETCDF_ROOT/lib:$HDF5_ROOT/lib
module list

