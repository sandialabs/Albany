#!/bin/csh

BASE_DIR=/home/projects/albany/nightlyCDashAlbanyBlake
cd $BASE_DIR

rm -rf gcc_modules.out 

unset http_proxy
unset https_proxy

source blake_gcc_modules.sh >& gcc_modules.out

export OMP_NUM_THREADS=1

LOG_FILE=$BASE_DIR/nightly_log_blakeAlbanyGccDebug.txt

bash convert-cmake-to-cdash-albany.sh gcc-debug
bash create-new-cdash-cmake-script-albany.sh gcc-debug

eval "env  TEST_DIRECTORY=$BASE_DIR SCRIPT_DIRECTORY=$BASE_DIR ctest -VV -S $BASE_DIR/ctest_nightly_albany-gcc-debug.cmake" > $LOG_FILE 2>&1

