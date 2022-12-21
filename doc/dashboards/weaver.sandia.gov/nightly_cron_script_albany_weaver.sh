#!/bin/csh

BASE_DIR=/home/projects/albany/nightlyCDashWeaver
cd $BASE_DIR


#source clean-up-no-trilinos.sh 

unset http_proxy
unset https_proxy

export jenkins_albany_dir=/home/projects/albany/nightlyCDashWeaver/repos/Albany
export jenkins_trilinos_dir=/home/projects/albany/nightlyCDashWeaver/repos/Trilinos

LOG_FILE=$BASE_DIR/nightly_log_weaverAlbany.txt

bash convert-cmake-to-cdash-albany.sh regular
bash create-new-cdash-cmake-script-albany.sh regular

eval "env  TEST_DIRECTORY=$BASE_DIR SCRIPT_DIRECTORY=$BASE_DIR ctest -VV -S $BASE_DIR/ctest_nightly_albany.cmake" > $LOG_FILE 2>&1

