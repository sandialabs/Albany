#!/bin/csh

BASE_DIR=/pscratch/sd/m/mcarlson/biweeklyCDashPerlmutter-serial
cd $BASE_DIR

unset http_proxy
unset https_proxy

export jenkins_albany_dir=${BASE_DIR}/repos/Albany
export jenkins_trilinos_dir=${BASE_DIR}/repos/Trilinos

source ${BASE_DIR}/convert-cmake-to-cdash.sh
source ${BASE_DIR}/create-new-cdash-cmake-script.sh

LOG_FILE=$BASE_DIR/biweekly_log_pm_cpu_Trilinos.txt

env TEST_DIRECTORY=$BASE_DIR SCRIPT_DIRECTORY=$BASE_DIR ctest -VV -S $BASE_DIR/ctest_nightly_trilinos.cmake > $LOG_FILE 2>&1

