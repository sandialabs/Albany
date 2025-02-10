#!/bin/csh

FAD_CONFIGURATION=${1}
FAD_SIZE=${2}

BASE_DIR=/pscratch/sd/m/mcarlson/biweeklyCDashPerlmutter-cuda
DEPLOY_DIR=/global/cfs/cdirs/fanssie/automated_testing/weeklyCDashPerlmutter/cuda
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

eval "BASE_DIR=${BASE_DIR} DEPLOY_DIR=${DEPLOY_DIR} FAD_CONFIGURATION=${FAD_CONFIGURATION} FAD_SIZE=${FAD_SIZE} TEST_DIRECTORY=$BASE_DIR SCRIPT_DIRECTORY=$BASE_DIR ctest -VV -S ${CMAKE_FILENAME}" > $LOG_FILE 2>&1

