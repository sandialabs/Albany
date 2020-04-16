#!/bin/csh

#BASE_DIR=/mscratch/albany/mayer/nightlyCDashJenkins
BASE_DIR=`pwd`
cd $BASE_DIR

unset http_proxy
unset https_proxy

export OMP_NUM_THREADS=1

LOG_FILE=$BASE_DIR/nightly_log_mayerAlbanySubmit.txt

now="$(date +'%Y%m%d')"
sed -i "s/XXX/$now/g" ctest_nightly_jenkins_albany_submit.cmake 


eval "env  TEST_DIRECTORY=$BASE_DIR SCRIPT_DIRECTORY=$BASE_DIR ctest -VV -S $BASE_DIR/ctest_nightly_jenkins_albany_submit.cmake" > $LOG_FILE 2>&1

