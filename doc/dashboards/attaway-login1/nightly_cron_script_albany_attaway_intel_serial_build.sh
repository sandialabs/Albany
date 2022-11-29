#!/bin/csh

BASE_DIR=/projects/albany/nightlyCDash
cd $BASE_DIR

unset http_proxy
unset https_proxy

source attaway_modules_intel.sh >& intel_modules.out  
cat build_albany.cmake ctest_nightly_albany_intel_serial_tmp.cmake >& ctest_nightly_albany_intel_serial_build.cmake

LOG_FILE=$BASE_DIR/nightly_log_attawayAlbanyIntelSerialBuild.txt

eval "env  TEST_DIRECTORY=$BASE_DIR SCRIPT_DIRECTORY=$BASE_DIR ctest -VV -S $BASE_DIR/ctest_nightly_albany_intel_serial_build.cmake" > $LOG_FILE 2>&1

