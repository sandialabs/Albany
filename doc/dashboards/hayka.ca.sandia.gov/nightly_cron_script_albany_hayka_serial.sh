#!/bin/csh

BASE_DIR=/home/ikalash/nightlyAlbanyCDash
cd $BASE_DIR

rm -rf hayka_modules.out 
rm -rf ctest_nightly_albany_serial.cmake

unset http_proxy
unset https_proxy

source modules_hayka.sh >& hayka_modules.out  

export OMP_NUM_THREADS=1

LOG_FILE=$BASE_DIR/nightly_log_haykaAlbany.txt

bash convert-cmake-to-cdash-albany.sh 
bash create-new-cdash-cmake-script-albany.sh 

eval "env  TEST_DIRECTORY=$BASE_DIR SCRIPT_DIRECTORY=$BASE_DIR ctest -VV -S $BASE_DIR/ctest_nightly_albany_serial.cmake" > $LOG_FILE 2>&1

