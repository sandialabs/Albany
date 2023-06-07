#!/bin/csh

BASE_DIR=/home/projects/albany/nightlyCDashTrilinosBlake
cd $BASE_DIR

rm -rf gcc_modules.out 

unset http_proxy
unset https_proxy


source blake_gcc_modules.sh >& gcc_modules.out  
cmake --version

source convert-cmake-to-cdash.sh gcc debug
source create-new-cdash-cmake-script.sh gcc debug

LOG_FILE=$BASE_DIR/nightly_log_blakeTrilinosGccDebug.txt

eval "env  TEST_DIRECTORY=$BASE_DIR SCRIPT_DIRECTORY=$BASE_DIR ctest -VV -S $BASE_DIR/ctest_nightly_trilinos_gcc_debug.cmake" > $LOG_FILE 2>&1

