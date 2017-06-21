#!/bin/bash
module purge
module load cmake git
module load openmpi/1.10.4/gcc/5.4.0/cuda/8.0.44
module load boost/1.60.0/openmpi/1.10.4/gcc/5.4.0/cuda/8.0.44
module load netcdf/4.4.1/openmpi/1.10.4/gcc/5.4.0/cuda/8.0.44
module load netcdf-f/4.4.4/openmpi/1.10.4/gcc/5.4.0/cuda/8.0.44 
module load pnetcdf/1.6.1/openmpi/1.10.4/gcc/5.4.0/cuda/8.0.44
module load hdf5/1.8.17/openmpi/1.10.4/gcc/5.4.0/cuda/8.0.44
module load openblas/0.2.19/gcc/5.4.0
module load hwloc/1.11.3/cuda/8.0.44
module list

#FIXME: the following needs needs to be changed to point to your Trilinos!
export OMPI_CXX=${HOME}/Trilinos/packages/kokkos/config/nvcc_wrapper
export CUDA_MANAGED_FORCE_DEVICE_ALLOC=1
export CUDA_LAUNCH_BLOCKING=1
