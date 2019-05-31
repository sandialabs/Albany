#!/bin/bash
module purge
module load git
module load devpack/20180521/openmpi/2.1.2/gcc/7.2.0/cuda/9.2.88
module swap cmake cmake/3.12.3
module list
#FIXME: the following needs needs to be changed to point to your Trilinos!
#export OMPI_CXX=${jenkins_trilinos_dir}/packages/kokkos/config/nvcc_wrapper
export CUDA_MANAGED_FORCE_DEVICE_ALLOC=1
export CUDA_LAUNCH_BLOCKING=1
 
