#!/bin/csh

BASE_DIR=/home/ikalash/nightlyAlbanyCDash
cd $BASE_DIR

rm -rf build
rm -rf repos
rm -rf slurm*
rm -rf *.txt
rm -rf hayka_modules.out 
rm -rf cdash*.txt
rm -rf ctest_nightly_trilinos_serial.cmake

unset http_proxy
unset https_proxy
unset HTTP_PROXY 
unset HTTPS_PROXY 

source modules_hayka.sh >& hayka_modules.out  
source convert-cmake-to-cdash.sh 
source create-new-cdash-cmake-script.sh 

LOG_FILE=$BASE_DIR/nightly_log_haykaTrilinos.txt

eval "env  TEST_DIRECTORY=$BASE_DIR SCRIPT_DIRECTORY=$BASE_DIR ctest -VV -S $BASE_DIR/ctest_nightly_trilinos_serial.cmake" > $LOG_FILE 2>&1

