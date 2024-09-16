#!/bin/csh

BASE_DIR=/lustre/orion/cli193/scratch/mcarlson/testingCDashFrontier-rocm
cd $BASE_DIR

unset all_proxy
unset ftp_proxy
unset http_proxy
unset https_proxy
unset no_proxy

source convert-cmake-to-cdash-albany.sh
source create-new-cdash-cmake-script-albany.sh

LOG_FILE=$BASE_DIR/build_log_frontier_Albany.txt

eval "env BUILD_OR_TEST=build env TEST_DIRECTORY=$BASE_DIR SCRIPT_DIRECTORY=$BASE_DIR ctest -VV -S $BASE_DIR/ctest_nightly_albany.cmake" > $LOG_FILE 2>&1

