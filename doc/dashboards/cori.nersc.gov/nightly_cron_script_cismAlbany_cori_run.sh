#!/bin/csh

BASE_DIR=/project/projectdirs/piscees/nightlyCoriCDash
EXE_DIR=/project/projectdirs/piscees/nightlyCoriCDashExe
cd $BASE_DIR

source cori_modules.sh >& modules.out 
export CRAYPE_LINK_TYPE=STATIC

LOG_FILE=$BASE_DIR/nightly_log_coriCismAlbanyRun.txt

eval "env  TEST_DIRECTORY=$BASE_DIR SCRIPT_DIRECTORY=$BASE_DIR ctest -VV -S $BASE_DIR/ctest_nightly_cismAlbany_run.cmake" > $LOG_FILE 2>&1

cp -r build/CoriCismAlbany/cism_driver/cism_driver $EXE_DIR
chmod -R 0755 $EXE_DIR

#bash process_output.sh 
#bash send_email.sh  
