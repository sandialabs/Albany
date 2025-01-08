#!/bin/csh

FAD_CONFIGURATION=${1}
FAD_SIZE=${2}

BASE_DIR=/pscratch/sd/m/mcarlson/biweeklyCDashPerlmutter-cuda-uvm
cd $BASE_DIR

source convert-cmake-to-cdash-albany.sh ${FAD_CONFIGURATION} ${FAD_SIZE}
source create-new-cdash-cmake-script-albany.sh ${FAD_CONFIGURATION} ${FAD_SIZE}

if [ "$FAD_CONFIGURATION" = "slfad" ] ; then
  LOG_FILE=$BASE_DIR/biweekly_log_pm_gpu_Albany_slfad.txt
  CMAKE_FILENAME=$BASE_DIR/ctest_nightly_albany_slfad.cmake
fi
if [ "$FAD_CONFIGURATION" = "sfad" ] && [ "$FAD_SIZE" = "12" ]; then
  LOG_FILE=$BASE_DIR/biweekly_log_pm_gpu_Albany_sfad12.txt
  CMAKE_FILENAME=$BASE_DIR/ctest_nightly_albany_sfad12.cmake
fi
if [ "$FAD_CONFIGURATION" = "sfad" ] && [ "$FAD_SIZE" = "24" ]; then
  LOG_FILE=$BASE_DIR/biweekly_log_pm_gpu_Albany_sfad24.txt
  CMAKE_FILENAME=$BASE_DIR/ctest_nightly_albany_sfad24.cmake
fi

eval "env TEST_DIRECTORY=$BASE_DIR SCRIPT_DIRECTORY=$BASE_DIR ctest -VV -S $BASE_DIR/ctest_nightly_albany.cmake" > $LOG_FILE 2>&1

