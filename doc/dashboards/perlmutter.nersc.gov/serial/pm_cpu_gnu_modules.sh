#!/bin/bash

module load PrgEnv-gnu/8.5.0
module load gcc-native/12.3
module load craype-accel-host
module load cray-libsci/23.12.5
module load craype/2.7.30
module load cray-mpich/8.1.28
module load cray-hdf5-parallel/1.12.2.9
module load cray-netcdf-hdf5parallel/4.9.0.9
module load cray-parallel-netcdf/1.12.3.9
module load cmake/3.30.2

export CRAYPE_LINK_TYPE=dynamic

export MPICH_ENV_DISPLAY=1
export MPICH_VERSION_DISPLAY=1
export OMP_STACKSIZE=128M
export OMP_PROC_BIND=spread
export OMP_PLACES=threads
export HDF5_USE_FILE_LOCKING=FALSE
export FI_CXI_RX_MATCH_MODE=software

export SUPERLU_DIR=/global/cfs/cdirs/fanssie/software/tpls/superlu-7.0.0

# Need this to avoid error when running seacas decomp
export LD_LIBRARY_PATH=${CRAY_LD_LIBRARY_PATH}:${LD_LIBRARY_PATH}

unset MPICH_GPU_SUPPORT_ENABLED