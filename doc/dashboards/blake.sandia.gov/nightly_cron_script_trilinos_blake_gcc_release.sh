#!/bin/csh

BASE_DIR=/home/projects/albany/nightlyCDashTrilinosBlake
cd $BASE_DIR

rm -rf build-gcc
rm -rf repos-gcc
rm -rf nightly*Gcc*.txt
rm -rf gcc_modules.out 
rm -rf cdash*gcc*.txt
rm -rf ctest*gcc*release.cmake

unset http_proxy
unset https_proxy


source blake_gcc_modules.sh >& gcc_modules.out  
cmake --version

source convert-cmake-to-cdash.sh gcc release
source create-new-cdash-cmake-script.sh gcc release

LOG_FILE=$BASE_DIR/nightly_log_blakeTrilinosGccRelease.txt

eval "env  TEST_DIRECTORY=$BASE_DIR SCRIPT_DIRECTORY=$BASE_DIR ctest -VV -S $BASE_DIR/ctest_nightly_trilinos_gcc_release.cmake" > $LOG_FILE 2>&1

