#!/bin/csh

BASE_DIR=/project/projectdirs/piscees/nightlyCoriCDash
EXE_DIR=/project/projectdirs/piscees/nightlyCoriCDashExe
cd $BASE_DIR

cp $BASE_DIR/repos/Albany/doc/dashboards/cori.nersc.gov/ctest_nightly_cismAlbany_submit.cmake $BASE_DIR

source cori_modules.sh >& modules.out 
#export CRAYPE_LINK_TYPE=STATIC

now="$(date +'%Y%m%d')"
sed -i "s/XXX/$now/g" ctest_nightly_cismAlbany_submit.cmake 

LOG_FILE=$BASE_DIR/nightly_log_coriCismAlbanySubmit.txt

eval "env  TEST_DIRECTORY=$BASE_DIR SCRIPT_DIRECTORY=$BASE_DIR ctest -VV -S $BASE_DIR/ctest_nightly_cismAlbany_submit.cmake" > $LOG_FILE 2>&1

