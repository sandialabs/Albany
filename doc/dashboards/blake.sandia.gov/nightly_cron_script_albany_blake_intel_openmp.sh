#!/bin/csh

BASE_DIR=/home/projects/albany/nightlyCDashAlbanyBlake
cd $BASE_DIR

rm -rf intel_modules.out 

unset http_proxy
unset https_proxy

export INTEL_LICENSE_FILE=/home/projects/x86-64/intel/licenses/USE_SERVER-ohpc.lic
source blake_intel_modules.sh >& intel_modules.out  

#export OMP_DISPLAY_ENV=TRUE
export OMP_NUM_THREADS=2
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

LOG_FILE=$BASE_DIR/nightly_log_blakeAlbanyIntelOpenMP.txt

bash convert-cmake-to-cdash-albany.sh intel-openmp
bash create-new-cdash-cmake-script-albany.sh intel-openmp

eval "env  TEST_DIRECTORY=$BASE_DIR SCRIPT_DIRECTORY=$BASE_DIR ctest -VV -S $BASE_DIR/ctest_nightly_albany-intel-openmp.cmake" > $LOG_FILE 2>&1

