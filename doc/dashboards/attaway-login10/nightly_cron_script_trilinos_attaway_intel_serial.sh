#!/bin/csh

BASE_DIR=/projects/albany/nightlyCDash
cd $BASE_DIR

rm -rf build
rm -rf repos
rm -rf slurm*
rm -rf *.txt
rm -rf intel_modules.out 
rm -rf cdash*intel*.txt
rm -rf ctest_nightly_trilinos_intel_serial.cmake
rm -rf ctest_nightly_albany_intel_serial_build.cmake

unset http_proxy
unset https_proxy
unset HTTP_PROXY
unset HTTPS_PROXY 

source attaway_modules_intel.sh >& intel_modules.out  
source convert-cmake-to-cdash.sh intel serial
source create-new-cdash-cmake-script.sh intel serial

LOG_FILE=$BASE_DIR/nightly_log_attawayTrilinosIntelSerial.txt

eval "env  TEST_DIRECTORY=$BASE_DIR SCRIPT_DIRECTORY=$BASE_DIR ctest -VV -S $BASE_DIR/ctest_nightly_trilinos_intel_serial.cmake" > $LOG_FILE 2>&1

