#!/bin/bash
module unload cray-hdf5-parallel
module unload cray-netcdf-hdf5parallel
module unload cray-parallel-netcdf
module unload PrgEnv-gnu
module unload PrgEnv-nvidia
module unload gpu
module unload cudatoolkit
module unload craype-accel-nvidia80
module unload craype-accel-host
module unload perftools-base
module unload perftools
module unload darshan

module load PrgEnv-gnu/8.3.3
module load gcc/11.2.0
module load craype-accel-host
module load cray-libsci/23.02.1.1
module load craype/2.7.20
module load cray-mpich/8.1.25
module load cray-hdf5-parallel/1.12.2.3
module load cray-netcdf-hdf5parallel/4.9.0.3
module load cray-parallel-netcdf/1.12.3.3
module load cmake/3.24.3

module load cray-python/3.9.13.1

module load e4s
spack env activate -V gcc
spack load superlu

export BOOST_DIR=/global/common/software/fanssie/boost-1.72.0/gcc-11.2.0

export MPICH_ENV_DISPLAY=1
export MPICH_VERSION_DISPLAY=1
export OMP_STACKSIZE=128M
export OMP_PROC_BIND=spread
export OMP_PLACES=threads
export HDF5_USE_FILE_LOCKING=FALSE
export FI_CXI_RX_MATCH_MODE=software
