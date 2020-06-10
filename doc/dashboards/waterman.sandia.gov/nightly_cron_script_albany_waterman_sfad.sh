#!/bin/csh

BASE_DIR=/home/projects/albany/waterman
cd $BASE_DIR

BUILD_OPT="$1"

if [ -z "$BUILD_OPT" ]; then
   echo "Please supply an argument: sfad4, sfad6, sfad8 or sfad12"
   exit 1;
fi

unset http_proxy
unset https_proxy

export jenkins_albany_dir=/home/projects/albany/waterman/repos/Albany
export jenkins_trilinos_dir=/home/projects/albany/waterman/repos/Trilinos

if [ "$BUILD_OPT" = "sfad4" ] ; then
  LOG_FILE=$BASE_DIR/nightly_log_watermanAlbanySFad4.txt
fi
if [ "$BUILD_OPT" = "sfad6" ] ; then
  LOG_FILE=$BASE_DIR/nightly_log_watermanAlbanySFad6.txt
fi
if [ "$BUILD_OPT" = "sfad8" ] ; then
  LOG_FILE=$BASE_DIR/nightly_log_watermanAlbanySFad8.txt
fi
if [ "$BUILD_OPT" = "sfad12" ] ; then
  LOG_FILE=$BASE_DIR/nightly_log_watermanAlbanySFad12.txt
fi

eval "env BUILD_OPTION=$BUILD_OPT TEST_DIRECTORY=$BASE_DIR SCRIPT_DIRECTORY=$BASE_DIR ctest -VV -S $BASE_DIR/ctest_nightly_albanySFAD.cmake" > $LOG_FILE 2>&1

