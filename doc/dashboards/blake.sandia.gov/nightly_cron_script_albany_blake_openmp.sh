#!/bin/csh

BASE_DIR=/home/projects/albany/nightlyCDashAlbanyBlake
cd $BASE_DIR

#rm -rf build
#rm -rf repos 
#rm -rf *log*
rm -rf results_blake_openmp
rm -rf ctest_nightly.cmake 
#rm -rf modules.out 

unset http_proxy
unset https_proxy

source blake_intel_modules.sh >& modules.out  

#export OMP_DISPLAY_ENV=TRUE
export OMP_NUM_THREADS=2
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

now=$(date +"%m_%d_%Y-%H_%M")
LOG_FILE=$BASE_DIR/nightly_log_blakeAlbanyOpenMP.txt

eval "env  TEST_DIRECTORY=$BASE_DIR SCRIPT_DIRECTORY=$BASE_DIR ctest -VV -S $BASE_DIR/ctest_nightly_albany_openmp.cmake" > $LOG_FILE 2>&1

bash process_results_ctest_openmp.sh
