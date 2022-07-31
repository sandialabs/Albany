#!/bin/csh

BASE_DIR=/home/projects/albany/nightlyCDashWeaver
cd $BASE_DIR

#source clean-up.sh 

unset http_proxy
unset https_proxy

export jenkins_albany_dir=/home/projects/albany/nightlyCDashWeaver/repos/Albany
export jenkins_trilinos_dir=/home/projects/albany/nightlyCDashWeaver/repos/Trilinos

source convert-cmake-to-cdash.sh
source create-new-cdash-cmake-script.sh

LOG_FILE=$BASE_DIR/nightly_log_weaverTrilinos.txt

eval "env  TEST_DIRECTORY=$BASE_DIR SCRIPT_DIRECTORY=$BASE_DIR ctest -VV -S $BASE_DIR/ctest_nightly_trilinos.cmake" > $LOG_FILE 2>&1

