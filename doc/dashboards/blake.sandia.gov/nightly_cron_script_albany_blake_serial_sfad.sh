#!/bin/bash

BASE_DIR=/home/projects/albany/nightlyCDashAlbanyBlake
cd $BASE_DIR

BUILD_OPT="$1"

if [ -z "$BUILD_OPT" ]; then
   echo "Please supply an argument: sfad4, sfad6, sfad8 or sfad12"
   exit 1;
fi

unset http_proxy
unset https_proxy

export INTEL_LICENSE_FILE=/home/projects/x86-64/intel/licenses/USE_SERVER-ohpc.lic
source blake_intel_modules.sh >& modules.out  

export OMP_NUM_THREADS=1

if [ "$BUILD_OPT" = "sfad4" ] ; then
  LOG_FILE=$BASE_DIR/nightly_log_blakeAlbanySerialSFad4.txt
fi
if [ "$BUILD_OPT" = "sfad6" ] ; then
  LOG_FILE=$BASE_DIR/nightly_log_blakeAlbanySerialSFad6.txt
fi
if [ "$BUILD_OPT" = "sfad8" ] ; then
  LOG_FILE=$BASE_DIR/nightly_log_blakeAlbanySerialSFad8.txt
fi
if [ "$BUILD_OPT" = "sfad12" ] ; then
  LOG_FILE=$BASE_DIR/nightly_log_blakeAlbanySerialSFad12.txt
fi

eval "env BUILD_OPTION=$BUILD_OPT TEST_DIRECTORY=$BASE_DIR SCRIPT_DIRECTORY=$BASE_DIR ctest -VV -S $BASE_DIR/ctest_nightly_albany_sfad.cmake" > $LOG_FILE 2>&1

