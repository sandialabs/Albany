#!/bin/csh

BASE_DIR=/home/projects/albany/nightlyCDashAlbanyBlake
cd $BASE_DIR


unset http_proxy
unset https_proxy

source blake_intel_modules.sh >& gcc_modules.out  

export OMP_NUM_THREADS=1

LOG_FILE=$BASE_DIR/nightly_log_blakeAlbanyGccSerial.txt

eval "env  TEST_DIRECTORY=$BASE_DIR SCRIPT_DIRECTORY=$BASE_DIR ctest -VV -S $BASE_DIR/ctest_nightly_albany_gcc_serial.cmake" > $LOG_FILE 2>&1

#bash process_results_ctest_serial.sh
