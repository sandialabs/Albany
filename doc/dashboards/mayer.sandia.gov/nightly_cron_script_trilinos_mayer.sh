#!/bin/csh

BASE_DIR=.
cd $BASE_DIR

rm -rf build
#rm -rf repos 
rm -rf *log*
rm -rf results_arm
rm -rf ctest_nightly.cmake 

unset http_proxy
unset https_proxy

#export OMP_NUM_THREADS=4
#export OMP_PLACES=threads
#export OMP_PROC_BIND=spread

cat trilinos ctest_nightly.cmake.frag >& ctest_nightly.cmake  

now=$(date +"%m_%d_%Y-%H_%M")
LOG_FILE=$BASE_DIR/nightly_log_mayerTrilinos.txt

eval "env  TEST_DIRECTORY=$BASE_DIR SCRIPT_DIRECTORY=$BASE_DIR ctest -VV -S $BASE_DIR/ctest_nightly.cmake" > $LOG_FILE 2>&1

