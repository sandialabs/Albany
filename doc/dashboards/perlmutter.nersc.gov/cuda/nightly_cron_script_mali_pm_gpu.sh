#!/bin/csh

BASE_DIR=/pscratch/sd/m/mcarlson/biweeklyCDashPerlmutter-cuda

unset http_proxy
unset https_proxy

LOG_FILE=$BASE_DIR/biweekly_log_pm_gpu_MALI.txt

eval "env TEST_DIRECTORY=$BASE_DIR SCRIPT_DIRECTORY=$BASE_DIR ctest -VV -S $BASE_DIR/ctest_nightly_mali.cmake" > $LOG_FILE 2>&1
