#!/bin/csh

BASE_DIR=/home/projects/albany/nightlyCDashTrilinosJenkins
cd $BASE_DIR

unset http_proxy
unset https_proxy

cat trilinos_jenkins ctest_nightly_jenkins.cmake.frag >& ctest_nightly.cmake  

now=$(date +"%m_%d_%Y-%H_%M")
LOG_FILE=$BASE_DIR/nightly_log_mayerTrilinos.txt

eval "env  TEST_DIRECTORY=$BASE_DIR SCRIPT_DIRECTORY=$BASE_DIR ctest -VV -S $BASE_DIR/ctest_nightly.cmake" > $LOG_FILE 2>&1

