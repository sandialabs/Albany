#!/bin/csh

BASE_DIR=/home/projects/albany/nightlyCDashTrilinosBlake
cd $BASE_DIR

export PATH=/home/projects/cmake-3.24.3/bin:$PATH


rm -rf build-gcc
rm -rf repos-gcc
rm -rf nightly*Gcc*.txt
rm -rf gcc_modules.out 
rm -rf cdash*gcc*.txt
rm -rf ctest*gcc*serial.cmake

unset http_proxy
unset https_proxy

#export OMP_DISPLAY_ENV=TRUE
#export OMP_NUM_THREADS=2
#export OMP_PLACES=threads
#export OMP_PROC_BIND=spread

#export INTEL_LICENSE_FILE=/home/projects/x86-64/intel/licenses/USE_SERVER-ohpc.lic
source blake_gcc_modules.sh >& gcc_modules.out  
source convert-cmake-to-cdash.sh gcc serial
source create-new-cdash-cmake-script.sh gcc serial

LOG_FILE=$BASE_DIR/nightly_log_blakeTrilinosGccSerial.txt

eval "env  TEST_DIRECTORY=$BASE_DIR SCRIPT_DIRECTORY=$BASE_DIR ctest -VV -S $BASE_DIR/ctest_nightly_trilinos_gcc_serial.cmake" > $LOG_FILE 2>&1

