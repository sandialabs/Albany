#!/bin/bash
module reset
module switch Core Core/24.07
module switch PrgEnv-cray PrgEnv-gnu/8.3.3
module switch gcc gcc/12.2.0
module load craype-accel-amd-gfx90a
module load rocm/5.4.0
module load cray-python/3.11.5
module load cray-libsci
module load cmake/3.27.9
module load subversion
module load git
module load zlib
module load libfabric/1.15.2.0
module load cray-hdf5-parallel/1.12.2.1
module load cray-netcdf-hdf5parallel/4.9.0.1
module load cray-parallel-netcdf/1.12.3.1

export MPICH_GPU_SUPPORT_ENABLED=1