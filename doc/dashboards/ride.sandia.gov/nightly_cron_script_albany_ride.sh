#!/bin/csh

BASE_DIR=/home/projects/albany/ride
cd $BASE_DIR

unset http_proxy
unset https_proxy

export OMPI_CXX=nvcc_wrapper_p100 

cat albany ctest_nightly.cmake.frag >& ctest_nightly.cmake  

now=$(date +"%m_%d_%Y-%H_%M")
LOG_FILE=$BASE_DIR/nightly_log_rideAlbany.txt

eval "env TEST_DIRECTORY=$BASE_DIR SCRIPT_DIRECTORY=$BASE_DIR  ctest -VV -S $BASE_DIR/ctest_nightly.cmake" > $LOG_FILE 2>&1

#bash process_results_ctest.sh
#bash send_email_ctest.sh

