#!/bin/bash
# This script loads the necessary modules and sets the necessary 
# environment variables on the Ride cluster for the compilation 
# of Trilinos and Albany/Aeras for GPU architectures.
#

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

#FIXME: The following needs to be changed to point to your Trilinos!
#FIXME: Change to 'default_arch="sm_60"' inside nvcc_wrapper file
export OMPI_CXX=/home/jwatkin/Trilinos/packages/kokkos/config/nvcc_wrapper
export CUDA_MANAGED_FORCE_DEVICE_ALLOC=1
export CUDA_LAUNCH_BLOCKING=1
