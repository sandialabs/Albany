#!/bin/bash

module load PrgEnv-gnu/8.7.0
module load gcc-native/14
module load craype-accel-host
module load cray-libsci/26.03.0
module load craype/2.7.36
module load cray-mpich/9.1.0
module load cray-hdf5-parallel/1.14.3.9
module load cray-netcdf-hdf5parallel/4.9.2.3
module load cray-parallel-netcdf/1.12.3.13
module load cmake/3.30.2

export CRAYPE_LINK_TYPE=dynamic

export MPICH_ENV_DISPLAY=1
export MPICH_VERSION_DISPLAY=1
export OMP_STACKSIZE=128M
export OMP_PROC_BIND=spread
export OMP_PLACES=threads
export HDF5_USE_FILE_LOCKING=FALSE
export FI_CXI_RX_MATCH_MODE=software

export SUPERLU_DIR=/global/common/software/fanssie/superlu/7.0.1/gcc/14.3
export BOOST_DIR=/global/common/software/fanssie/boost/1.83.0/gcc/14.3

# Need this to avoid error when running seacas decomp
export LD_LIBRARY_PATH=${CRAY_LD_LIBRARY_PATH}:${LD_LIBRARY_PATH}

unset MPICH_GPU_SUPPORT_ENABLED
