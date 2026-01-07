#!/bin/bash
module reset
module load Core/25.03
module load PrgEnv-gnu
module load cpe/24.11
module load gcc-native/12.3
module load libunwind
module load cray-python/3.11.7
module load subversion
module load git
module load cmake
module load cray-hdf5-parallel/1.14.3.3
module load cray-netcdf-hdf5parallel/4.9.0.15
module load cray-parallel-netcdf/1.12.3.15
module unload darshan-runtime

module load craype-accel-amd-gfx90a
module load rocm/6.2.4

# Frontier's boost module is not compatible with our modules.
# A pre-built version of boost is available here for members of cli193,
# otherwise, you need to supply boost header files yourself
export BOOST_ROOT=/lustre/orion/cli193/proj-shared/automated_testing/boost_1_86_0

# Need this to avoid error when running seacas decomp
export LD_LIBRARY_PATH=${CRAY_LD_LIBRARY_PATH}:${LD_LIBRARY_PATH}

export MPICH_GPU_SUPPORT_ENABLED=1
