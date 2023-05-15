#!/bin/bash
module purge 
module load gcc/8.3.1
module load openmpi/4.1.4
module load cuda/11.8.0
module load git/2.39.1
module load cmake/3.25.1
module load python/3.10.8
module load openblas/0.3.23
module load metis/5.1.0
module load zlib/1.2.13
module load hdf5/1.12.2
module load parallel-netcdf/1.12.3
module load netcdf-c/4.9.0
module load parmetis/4.0.3
module load boost/1.80.0
module load superlu/5.3.0
module list

export CUDA_MANAGED_FORCE_DEVICE_ALLOC=1
export CUDA_LAUNCH_BLOCKING=1
 
