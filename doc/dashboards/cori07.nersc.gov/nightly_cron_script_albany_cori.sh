#!/bin/csh

BASE_DIR=/project/projectdirs/piscees/nightlyCoriCDash
EXE_DIR=/project/projectdirs/piscees/nightlyCoriCDashExe

cd $BASE_DIR

source cori_modules.sh >& modules.out 
#export CRAYPE_LINK_TYPE=STATIC

#IKT, 3/19/2022: older cmake is needed to be able to push to CDash site
export PATH=/project/projectdirs/piscees/tpl/cmake-3.18.0/bin:$PATH
cmake --version >& cmake_version_albany.out

LOG_FILE=$BASE_DIR/nightly_log_coriAlbany.txt

bash convert-cmake-to-cdash-albany.sh regular
bash create-new-cdash-cmake-script-albany.sh regular

eval "env  TEST_DIRECTORY=$BASE_DIR SCRIPT_DIRECTORY=$BASE_DIR ctest -VV -S $BASE_DIR/ctest_nightly_albany.cmake" > $LOG_FILE 2>&1

cp -r build/AlbanyInstall/bin/* $EXE_DIR
chmod -R 0755 $EXE_DIR

