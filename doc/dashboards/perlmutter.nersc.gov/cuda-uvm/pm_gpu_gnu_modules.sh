#!/bin/bash

module load cpe/23.03
module load PrgEnv-gnu/8.3.3
module load gcc/11.2.0
module load cudatoolkit/11.7
module load craype-accel-nvidia80
module load cray-libsci/23.02.1.1
module load craype/2.7.20
module load cray-mpich/8.1.25
module load cray-hdf5-parallel/1.12.2.3
module load cray-netcdf-hdf5parallel/4.9.0.3
module load cray-parallel-netcdf/1.12.3.3
module load cmake/3.24.3

module load e4s/23.05
spack env activate -V gcc
spack load superlu

export MPICH_ENV_DISPLAY=1
export MPICH_VERSION_DISPLAY=1
export OMP_STACKSIZE=128M
export OMP_PROC_BIND=spread
export OMP_PLACES=threads
export HDF5_USE_FILE_LOCKING=FALSE
export MPICH_GPU_SUPPORT_ENABLED=1
export CUDATOOLKIT_VERSION_STRING=${CRAY_CUDATOOLKIT_VERSION#*_}

export CUDA_MANAGED_FORCE_DEVICE_ALLOC=1
export TPETRA_ASSUME_GPU_AWARE_MPI=1
export FI_HMEM_CUDA_USE_GDRCOPY=0