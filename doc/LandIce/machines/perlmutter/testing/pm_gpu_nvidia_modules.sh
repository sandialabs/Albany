#!/bin/bash
module unload cray-hdf5-parallel
module unload cray-netcdf-hdf5parallel
module unload cray-parallel-netcdf
module unload PrgEnv-gnu
module unload PrgEnv-nvidia
module unload cudatoolkit
module unload craype-accel-nvidia80
module unload craype-accel-host
module unload perftools-base
module unload perftools
module unload darshan

module load PrgEnv-nvidia
module load nvidia/21.11
module load cudatoolkit/11.5
module load craype-accel-nvidia80
module load craype-accel-host
module load cray-libsci
module load craype
module load cray-mpich/8.1.15
module load cray-hdf5-parallel/1.12.1.1
module load cray-netcdf-hdf5parallel/4.8.1.1
module load cray-parallel-netcdf/1.12.2.1
module load cmake/3.22.0
module load boost

export MPICH_ENV_DISPLAY=1
export MPICH_VERSION_DISPLAY=1
export OMP_STACKSIZE=128M
export OMP_PROC_BIND=spread
export OMP_PLACES=threads
export HDF5_USE_FILE_LOCKING=FALSE
export MPICH_GPU_SUPPORT_ENABLED=1
