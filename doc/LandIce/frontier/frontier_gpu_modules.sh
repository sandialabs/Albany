#!/bin/bash
module reset
module switch PrgEnv-cray PrgEnv-gnu/8.3.3
module switch gcc gcc/11.2.0
module load craype-accel-amd-gfx90a
module load rocm/5.4.3
module load cray-python/3.9.13.1
module load subversion/1.14.1
module load git/2.36.1
module load zlib/1.2.11
module load cray-hdf5-parallel/1.12.2.1
module load cray-netcdf-hdf5parallel/4.9.0.1
module load cray-parallel-netcdf/1.12.3.1
module load boost/1.79.0
module load cmake/3.23.2

export MPICH_GPU_SUPPORT_ENABLED=1
