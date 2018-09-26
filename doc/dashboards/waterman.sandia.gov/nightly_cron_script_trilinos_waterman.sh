#!/bin/csh

BASE_DIR=/home/projects/albany/waterman
cd $BASE_DIR

rm -rf repos
rm -rf build
rm -rf ctest_nightly.cmake
rm -rf nightly_log*
rm -rf results*
rm -rf modules*out 

unset http_proxy
unset https_proxy

cat trilinos ctest_nightly.cmake.frag >& ctest_nightly.cmake  

export jenkins_albany_dir=/home/projects/albany/waterman/repos/Albany
export jenkins_trilinos_dir=/home/projects/albany/waterman/repos/Trilinos

now=$(date +"%m_%d_%Y-%H_%M")
LOG_FILE=$BASE_DIR/nightly_log_watermanTrilinos.txt

eval "env  TEST_DIRECTORY=$BASE_DIR SCRIPT_DIRECTORY=$BASE_DIR ctest -VV -S $BASE_DIR/ctest_nightly.cmake" > $LOG_FILE 2>&1

