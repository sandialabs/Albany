#!/bin/csh

BASE_DIR=/pscratch/sd/j/jwatkins/nightlyCDashPerlmutterCuda
cd $BASE_DIR

unset http_proxy
unset https_proxy

source convert-cmake-to-cdash.sh
source create-new-cdash-cmake-script.sh

LOG_FILE=$BASE_DIR/biweekly_log_pm_gpu_Trilinos.txt

eval "TEST_DIRECTORY=$BASE_DIR SCRIPT_DIRECTORY=$BASE_DIR ctest -VV -S $BASE_DIR/ctest_nightly_trilinos.cmake" > $LOG_FILE 2>&1

