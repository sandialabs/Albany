#!/bin/csh

BASE_DIR=/home/ikalash/nightlyCDash
cd $BASE_DIR

module purge
module load cmake/2.8.11.2
module load openmpi/1.10.1/gnu/4.7.2/cuda/7.5.7
module load intel/15.5.223
module load nvcc-wrapper/gnu
export CC=mpicc
export CXX=mpicxx
export FC=mpif90
export NVCC_WRAPPER_DEFAULT_COMPILER=mpicc
#FIXME: the following needs needs to be changed to point to your Trilinos! 
export OMPI_CXX=/home/jwatkin/Trilinos/packages/kokkos/config/nvcc_wrapper
export CUDA_MANAGED_FORCE_DEVICE_ALLOC=1
export CUDA_LAUNCH_BLOCKING=1

rm -rf $BASE_DIR/repos
rm -rf $BASE_DIR/build
rm -rf $BASE_DIR/ctest_nightly.cmake.work
rm -rf $BASE_DIR/nightly_log*
rm -rf $BASE_DIR/results*
rm -rf $BASE_DIR/test_summary.txt

cat trilinosBuild ctest_nightly.cmake.frag >& ctest_nightly.cmake  

now=$(date +"%m_%d_%Y-%H_%M")
LOG_FILE=$BASE_DIR/nightly_log_shannonTrilinosBuild.txt

eval "env  TEST_DIRECTORY=$BASE_DIR SCRIPT_DIRECTORY=$BASE_DIR ctest -VV -S $BASE_DIR/ctest_nightly.cmake" > $LOG_FILE 2>&1

