#!/bin/csh

BASE_DIR=/home/projects/albany/ride
cd $BASE_DIR

unset http_proxy
unset https_proxy

now=$(date +"%m_%d_%Y-%H_%M")
LOG_FILE=$BASE_DIR/nightly_log_rideTrilinos.txt

eval "env  TEST_DIRECTORY=$BASE_DIR SCRIPT_DIRECTORY=$BASE_DIR ctest -VV -S $BASE_DIR/ctest_nightly_trilinos.cmake" > $LOG_FILE 2>&1

