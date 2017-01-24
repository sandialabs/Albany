#!/bin/csh

BASE_DIR=/home/ikalash/nightlyCDash
cd $BASE_DIR

unset http_proxy
unset https_proxy

cat trilinos ctest_nightly.cmake.frag >& ctest_nightly.cmake  

now=$(date +"%m_%d_%Y-%H_%M")
LOG_FILE=$BASE_DIR/nightly_log_rideTrilinos.txt

eval "env  TEST_DIRECTORY=$BASE_DIR SCRIPT_DIRECTORY=$BASE_DIR ctest -VV -S $BASE_DIR/ctest_nightly.cmake" > $LOG_FILE 2>&1

