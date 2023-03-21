#!/bin/csh

BASE_DIR=/home/projects/albany/nightlyCDashTrilinosBlake
cd $BASE_DIR

rm -rf build-intel
rm -rf repos-intel
rm -rf nightly*Intel*.txt
rm -rf intel_modules.out 
rm -rf cdash*intel*.txt
rm -rf ctest*intel*serial.cmake
rm -rf ctest*intel*openmp.cmake

unset http_proxy
unset https_proxy

export OMP_NUM_THREADS=1
export INTEL_LICENSE_FILE=/home/projects/x86-64/intel/licenses/USE_SERVER-ohpc.lic

source blake_intel_modules.sh >& intel_modules.out  
cmake --version

source convert-cmake-to-cdash.sh intel serial
source create-new-cdash-cmake-script.sh intel serial

LOG_FILE=$BASE_DIR/nightly_log_blakeTrilinosIntelSerial.txt

eval "env  TEST_DIRECTORY=$BASE_DIR SCRIPT_DIRECTORY=$BASE_DIR ctest -VV -S $BASE_DIR/ctest_nightly_trilinos_intel_serial.cmake" > $LOG_FILE 2>&1

