#!/bin/csh

BASE_DIR=/home/projects/albany/nightlyCDashTrilinosBlake
cd $BASE_DIR

rm -rf intel_modules.out 

unset http_proxy
unset https_proxy

export PATH=/home/projects/cmake-3.24.3/bin:$PATH

#export OMP_DISPLAY_ENV=TRUE
export OMP_NUM_THREADS=2
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

export INTEL_LICENSE_FILE=/home/projects/x86-64/intel/licenses/USE_SERVER-ohpc.lic
source blake_intel_modules.sh >& intel_modules.out  
source convert-cmake-to-cdash.sh intel openmp
source create-new-cdash-cmake-script.sh intel openmp

LOG_FILE=$BASE_DIR/nightly_log_blakeTrilinosIntelOpenMP.txt

eval "env  TEST_DIRECTORY=$BASE_DIR SCRIPT_DIRECTORY=$BASE_DIR ctest -VV -S $BASE_DIR/ctest_nightly_trilinos_intel_openmp.cmake" > $LOG_FILE 2>&1

