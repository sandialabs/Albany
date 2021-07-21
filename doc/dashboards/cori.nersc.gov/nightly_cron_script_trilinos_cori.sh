#!/bin/csh

BASE_DIR=/project/projectdirs/piscees/nightlyCoriCDash

cd $BASE_DIR

#export CRAYPE_LINK_TYPE=STATIC
source cori_modules.sh >& modules.out 

rm -rf $BASE_DIR/repos
rm -rf $BASE_DIR/build
rm -rf $BASE_DIR/ctest_nightly.cmake.work
rm -rf $BASE_DIR/nightly_log*
rm -rf $BASE_DIR/*out 
rm -rf $BASE_DIR/slurm* 
rm -rf $BASE_DIR/results*
rm -rf $BASE_DIR/test_summary.txt
rm -rf $BASE_DIR/ctest_nightly_cismAlbany_submit.cmake

LOG_FILE=$BASE_DIR/nightly_log_coriTrilinos.txt

source clean-up.sh 
source convert-cmake-to-cdash.sh            
source create-new-cdash-cmake-script.sh   

eval "env  TEST_DIRECTORY=$BASE_DIR SCRIPT_DIRECTORY=$BASE_DIR ctest -VV -S $BASE_DIR/ctest_nightly_trilinos.cmake" > $LOG_FILE 2>&1

