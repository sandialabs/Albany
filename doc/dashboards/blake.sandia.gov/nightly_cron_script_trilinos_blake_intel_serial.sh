#!/bin/csh

BASE_DIR=/home/projects/albany/nightlyCDashTrilinosBlake
cd $BASE_DIR

rm -rf build
rm -rf repos 
rm -rf *log*
rm -rf results_blake_serial
rm -rf ctest_nightly.cmake 
rm -rf intel_modules.out 

unset http_proxy
unset https_proxy

export OMP_NUM_THREADS=1
export INTEL_LICENSE_FILE=/home/projects/x86-64/intel/licenses/USE_SERVER-ohpc.lic

source clean-up.sh 
source blake_intel_modules.sh >& intel_modules.out  
source convert-cmake-to-cdash.sh intel serial
source create-new-cdash-cmake-script.sh intel serial

LOG_FILE=$BASE_DIR/nightly_log_blakeTrilinosIntelSerial.txt

eval "env  TEST_DIRECTORY=$BASE_DIR SCRIPT_DIRECTORY=$BASE_DIR ctest -VV -S $BASE_DIR/ctest_nightly_trilinos_intel_serial.cmake" > $LOG_FILE 2>&1

