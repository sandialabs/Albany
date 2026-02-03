#!/bin/csh

BASE_DIR=/pscratch/sd/j/jwatkins/nightlyCDashPerlmutterSerial

unset http_proxy
unset https_proxy

LOG_FILE=$BASE_DIR/biweekly_log_pm_cpu_MALI.txt

eval "env TEST_DIRECTORY=$BASE_DIR SCRIPT_DIRECTORY=$BASE_DIR ctest -VV -S $BASE_DIR/ctest_nightly_mali.cmake" > $LOG_FILE 2>&1
