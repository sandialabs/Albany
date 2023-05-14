#!/bin/bash
module purge 
module load git/2.31.1 cmake/3.21.2
module load cuda/11.2.2/gcc/8.3.1
module load openmpi/4.1.1/gcc/8.3.1/cuda/11.2.2
module load openblas/0.3.18/gcc/8.3.1
module load metis/5.1.0/gcc/8.3.1
module load zlib/1.2.11/gcc/8.3.1
module load hdf5/1.10.7/gcc/8.3.1/openmpi/4.1.1
module load parallel-netcdf/1.12.2/gcc/8.3.1/openmpi/4.1.1
module load netcdf-c/4.8.1/gcc/8.3.1/openmpi/4.1.1
module load parmetis/4.0.3/gcc/8.3.1/openmpi/4.1.1
module load boost/1.70.0/gcc/8.3.1
module load superlu/5.3.0/gcc/8.3.1
module load ucx/1.12.1/gcc/8.3.1
module list

export CUDA_MANAGED_FORCE_DEVICE_ALLOC=1
export CUDA_LAUNCH_BLOCKING=1
 
