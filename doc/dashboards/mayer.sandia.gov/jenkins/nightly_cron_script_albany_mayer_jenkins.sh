#!/bin/csh

BASE_DIR=/mscratch/albany/mayer/nightlyCDashJenkins
cd $BASE_DIR

unset http_proxy
unset https_proxy

export OMP_NUM_THREADS=1
#export OMP_NUM_THREADS=4
#export OMP_PLACES=threads
#export OMP_PROC_BIND=spread

now=$(date +"%m_%d_%Y-%H_%M")
LOG_FILE=$BASE_DIR/nightly_log_mayerAlbany.txt

eval "env  TEST_DIRECTORY=$BASE_DIR SCRIPT_DIRECTORY=$BASE_DIR ctest -VV -S $BASE_DIR/ctest_nightly_jenkins_albany.cmake" > $LOG_FILE 2>&1

bash process_results_ctest.sh
