#!/bin/csh

BASE_DIR=/home/projects/albany/nightlyCDashAlbanyBlake
cd $BASE_DIR

rm -rf build
rm -rf repos 
rm -rf *log*
rm -rf results_blake_serial
rm -rf ctest_nightly.cmake 
rm -rf modules.out 

unset http_proxy
unset https_proxy

export INTEL_LICENSE_FILE=/home/projects/x86-64/intel/licenses/USE_SERVER-ohpc.lic
source blake_intel_modules_submit.sh >& modules.out  

export OMP_NUM_THREADS=1

LOG_FILE=$BASE_DIR/nightly_log_blakeAlbanyIntelSerial.txt

eval "env  TEST_DIRECTORY=$BASE_DIR SCRIPT_DIRECTORY=$BASE_DIR ctest -VV -S $BASE_DIR/ctest_nightly_albany_intel_serial.cmake" > $LOG_FILE 2>&1

bash process_results_ctest_serial.sh
