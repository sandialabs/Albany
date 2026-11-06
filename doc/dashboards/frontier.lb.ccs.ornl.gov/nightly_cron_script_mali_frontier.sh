#!/bin/csh

BASE_DIR=/lustre/orion/cli193/scratch/jwatkins/nightlyCDashFrontier
DEPLOY_DIR=/lustre/orion/cli193/proj-shared/automated_testing/rocm
cd $BASE_DIR

unset all_proxy
unset ftp_proxy
unset http_proxy
unset https_proxy
unset no_proxy

LOG_FILE=$BASE_DIR/build_log_frontier_MALI.txt

eval "DEPLOY_DIR=${DEPLOY_DIR} TEST_DIRECTORY=$BASE_DIR SCRIPT_DIRECTORY=$BASE_DIR ctest -VV -S $BASE_DIR/ctest_nightly_mali.cmake" > $LOG_FILE 2>&1

