#!/bin/csh

BASE_DIR=/home/projects/albany/nightlyCDashWeaver
cd $BASE_DIR

BUILD_OPT="$1"

if [ -z "$BUILD_OPT" ]; then
   echo "Please supply an argument: sfad6, sfad12 or sfad24"
   exit 1;
fi

unset http_proxy
unset https_proxy

export jenkins_albany_dir=/home/projects/albany/nightlyCDashWeaver/repos/Albany
export jenkins_trilinos_dir=/home/projects/albany/nightlyCDashWeaver/repos/Trilinos

bash convert-cmake-to-cdash-albany.sh sfad
bash create-new-cdash-cmake-script-albany.sh sfad

if [ "$BUILD_OPT" = "sfad6" ] ; then
  LOG_FILE=$BASE_DIR/nightly_log_weaverAlbanySFad6.txt
fi
if [ "$BUILD_OPT" = "sfad12" ] ; then
  LOG_FILE=$BASE_DIR/nightly_log_weaverAlbanySFad12.txt
fi
if [ "$BUILD_OPT" = "sfad24" ] ; then
  LOG_FILE=$BASE_DIR/nightly_log_weaverAlbanySFad24.txt
fi

eval "env BUILD_OPTION=$BUILD_OPT TEST_DIRECTORY=$BASE_DIR SCRIPT_DIRECTORY=$BASE_DIR ctest -VV -S $BASE_DIR/ctest_nightly_albanySFAD.cmake" > $LOG_FILE 2>&1

