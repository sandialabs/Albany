#!/bin/csh

BASE_DIR=/home/projects/albany/nightlyCDashTrilinosBlake
cd $BASE_DIR

rm -rf build-intel
rm -rf repos-intel
rm -rf nightly*Intel*.txt
rm -rf intel_modules.out 
rm -rf cdash*intel*.txt
rm -rf ctest*intel*release.cmake

unset http_proxy
unset https_proxy


source blake_intel_modules.sh >& intel_modules.out  
cmake --version

source convert-cmake-to-cdash.sh intel release
source create-new-cdash-cmake-script.sh intel release

LOG_FILE=$BASE_DIR/nightly_log_blakeTrilinosIntelRelease.txt

eval "env  TEST_DIRECTORY=$BASE_DIR SCRIPT_DIRECTORY=$BASE_DIR ctest -VV -S $BASE_DIR/ctest_nightly_trilinos_intel_release.cmake" > $LOG_FILE 2>&1

