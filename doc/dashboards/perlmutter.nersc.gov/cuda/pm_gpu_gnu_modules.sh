#!/bin/bash

module load PrgEnv-gnu/8.5.0
module load gcc-native/12.3
module load cudatoolkit/12.9
module load craype-accel-nvidia80
module load cray-libsci/24.07.0
module load craype/2.7.32
module load cray-mpich/8.1.30
module load cray-hdf5-parallel/1.14.3.1
module load cray-netcdf-hdf5parallel/4.9.0.13
module load cray-parallel-netcdf/1.12.3.13
module load cmake/3.30.2

export CRAYPE_LINK_TYPE=dynamic

export MPICH_ENV_DISPLAY=1
export MPICH_VERSION_DISPLAY=1
export OMP_STACKSIZE=128M
export OMP_PROC_BIND=spread
export OMP_PLACES=threads
export HDF5_USE_FILE_LOCKING=FALSE
export MPICH_GPU_SUPPORT_ENABLED=1
export CUDATOOLKIT_VERSION_STRING=${CRAY_CUDATOOLKIT_VERSION#*_}

export KOKKOS_MAP_DEVICE_ID_BY=mpi_rank
export CUDA_MANAGED_FORCE_DEVICE_ALLOC=1
export TPETRA_ASSUME_GPU_AWARE_MPI=0

export BOOST_DIR=/global/common/software/fanssie/boost-1.72.0/gcc-12.3.0

# Need this to avoid error when running seacas decomp
export LD_LIBRARY_PATH=${CRAY_LD_LIBRARY_PATH}:${LD_LIBRARY_PATH}

# To enable GPU-aware MPI, use the following two environment variables instead
#export TPETRA_ASSUME_GPU_AWARE_MPI=1
#export FI_HMEM_CUDA_USE_GDRCOPY=0