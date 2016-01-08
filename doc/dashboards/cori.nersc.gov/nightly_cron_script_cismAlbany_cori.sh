#!/bin/csh

BASE_DIR=/project/projectdirs/piscees/nightlyCoriCDash
EXE_DIR=/project/projectdirs/piscees/nightlyCoriCDashExe
cd $BASE_DIR

source cori_modules.sh 

rm -rf $BASE_DIR/ctest_nightly.cmake.work

cat cismAlbanyFELIX ctest_nightly.cmake.frag >& ctest_nightly.cmake  


now=$(date +"%m_%d_%Y-%H_%M")
LOG_FILE=$BASE_DIR/nightly_log_coriCismAlbany.txt

eval "env  TEST_DIRECTORY=$BASE_DIR SCRIPT_DIRECTORY=$BASE_DIR ctest -VV -S $BASE_DIR/ctest_nightly.cmake" > $LOG_FILE 2>&1

cp -r build/CoriCismAlbany/cism_driver/cism_driver $EXE_DIR
chmod -R 0755 $EXE_DIR

bash process_output.sh 
bash send_email.sh  
