#!/bin/csh

BASE_DIR=/lustre/orion/cli193/scratch/mcarlson/testingCDashFrontier-rocm
DEPLOY_DIR=/lustre/orion/cli193/proj-shared/automated_testing/rocm
cd $BASE_DIR

unset all_proxy
unset ftp_proxy
unset http_proxy
unset https_proxy
unset no_proxy

source convert-cmake-to-cdash-albany.sh ${FAD_CONFIGURATION} ${FAD_SIZE}
source create-new-cdash-cmake-script-albany.sh ${FAD_CONFIGURATION} ${FAD_SIZE}

if [ "$FAD_CONFIGURATION" = "slfad" ] ; then
  LOG_FILE=$BASE_DIR/biweekly_log_frontier_Albany_slfad.txt
  CMAKE_FILENAME=$BASE_DIR/ctest_nightly_albany_slfad.cmake
fi
if [ "$FAD_CONFIGURATION" = "sfad" ] && [ "$FAD_SIZE" = "12" ]; then
  LOG_FILE=$BASE_DIR/biweekly_log_frontier_Albany_sfad12.txt
  CMAKE_FILENAME=$BASE_DIR/ctest_nightly_albany_sfad12.cmake
fi
if [ "$FAD_CONFIGURATION" = "sfad" ] && [ "$FAD_SIZE" = "24" ]; then
  LOG_FILE=$BASE_DIR/biweekly_log_frontier_Albany_sfad24.txt
  CMAKE_FILENAME=$BASE_DIR/ctest_nightly_albany_sfad24.cmake
fi

eval "BUILD_OR_TEST=build FAD_CONFIGURATION=${FAD_CONFIGURATION} FAD_SIZE=${FAD_SIZE} BASE_DIR=${BASE_DIR} DEPLOY_DIR=${DEPLOY_DIR} TEST_DIRECTORY=$BASE_DIR SCRIPT_DIRECTORY=$BASE_DIR ctest -VV -S ${CMAKE_FILENAME}" > $LOG_FILE 2>&1

