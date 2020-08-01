#!/bin/csh

BASE_DIR=/home/projects/albany/nightlyCDashTrilinosBlake
cd $BASE_DIR

#rm -rf build
#rm -rf repos 
#rm -rf *log*
rm -rf results_blake_openmp
rm -rf ctest_nightly.cmake 
rm -rf modules.out 

unset http_proxy
unset https_proxy

#export OMP_DISPLAY_ENV=TRUE
export OMP_NUM_THREADS=2
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

export INTEL_LICENSE_FILE=/home/projects/x86-64/intel/licenses/USE_SERVER-ohpc.lic
source blake_intel_modules.sh >& modules.out  
source convert-cmake-to-cdash.sh openmp
source create-new-cdash-cmake-script.sh openmp

LOG_FILE=$BASE_DIR/nightly_log_blakeTrilinosOpenMP.txt

eval "env  TEST_DIRECTORY=$BASE_DIR SCRIPT_DIRECTORY=$BASE_DIR ctest -VV -S $BASE_DIR/ctest_nightly_trilinos_openmp.cmake" > $LOG_FILE 2>&1

