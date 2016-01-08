#!/bin/csh

BASE_DIR=/project/projectdirs/piscees/nightlyEdisonCDash
cd $BASE_DIR

source edison_modules.sh 

rm -rf $BASE_DIR/repos
rm -rf $BASE_DIR/build
rm -rf $BASE_DIR/ctest_nightly.cmake.work
rm -rf $BASE_DIR/nightly_log*
rm -rf $BASE_DIR/results*
rm -rf $BASE_DIR/test_summary.txt

cat trilinosFELIX ctest_nightly.cmake.frag >& ctest_nightly.cmake  

now=$(date +"%m_%d_%Y-%H_%M")
LOG_FILE=$BASE_DIR/nightly_log_edisonTrilinos.txt

eval "env  TEST_DIRECTORY=$BASE_DIR SCRIPT_DIRECTORY=$BASE_DIR ctest -VV -S $BASE_DIR/ctest_nightly.cmake" > $LOG_FILE 2>&1

