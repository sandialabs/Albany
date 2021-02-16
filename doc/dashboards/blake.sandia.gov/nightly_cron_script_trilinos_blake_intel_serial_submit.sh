#!/bin/csh

BASE_DIR=/home/projects/albany/nightlyCDashTrilinosBlake
cd $BASE_DIR

cp $BASE_DIR/repos/Albany/doc/dashboards/blake.sandia.gov/ctest_nightly_trilinos_blake_intel_serial_submit.cmake $BASE_DIR

source blake_intel_modules_submit.sh >& intel_modules_submit.out 

now="$(date +'%Y%m%d')"
path1="$BASE_DIR/build/Testing/$now-0100" 
path2="$BASE_DIR/build/Testing/$now-0100-Intel-GCC"
mv $path1 $path2

sed -i "s/XXX/$now/g" ctest_nightly_trilinos_blake_intel_serial_submit.cmake 

LOG_FILE=$BASE_DIR/nightly_log_blakeTrilinosIntelSerial_submit.txt

eval "env  TEST_DIRECTORY=$BASE_DIR SCRIPT_DIRECTORY=$BASE_DIR ctest -VV -S $BASE_DIR/ctest_nightly_trilinos_blake_intel_serial_submit.cmake" > $LOG_FILE 2>&1

