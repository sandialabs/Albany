#!/bin/bash

BASE_DIR=/project/projectdirs/aeras/nightlyGaiaCDash
cd $BASE_DIR

source gaia_modules.sh 

rm -rf $BASE_DIR/repos
rm -rf $BASE_DIR/build
rm -rf $BASE_DIR/ctest_nightly.cmake.work
rm -rf $BASE_DIR/nightly_log*
rm -rf $BASE_DIR/results*
rm -rf $BASE_DIR/test_summary.txt

cat trilinosAeras ctest_nightly.cmake.frag >& ctest_nightly.cmake  

now=$(date +"%m_%d_%Y-%H_%M")
LOG_FILE=$BASE_DIR/nightly_log_gaiaTrilinos.txt

JOBS=12

eval "env  TEST_DIRECTORY=$BASE_DIR SCRIPT_DIRECTORY=$BASE_DIR ctest -VV -j $JOBS -S $BASE_DIR/ctest_nightly.cmake" > $LOG_FILE 2>&1

