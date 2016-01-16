#!/bin/csh

BASE_DIR=/project/projectdirs/piscees/nightlyEdisonCDash
EXE_DIR=/project/projectdirs/piscees/nightlyEdisonCDashExe
cd $BASE_DIR

source edison_modules.sh 

rm -rf $BASE_DIR/ctest_nightly.cmake.work

cat albanyFELIX ctest_nightly.cmake.frag >& ctest_nightly.cmake  

now=$(date +"%m_%d_%Y-%H_%M")
LOG_FILE=$BASE_DIR/nightly_log_edisonAlbany.txt

eval "env  TEST_DIRECTORY=$BASE_DIR SCRIPT_DIRECTORY=$BASE_DIR ctest -VV -S $BASE_DIR/ctest_nightly.cmake" > $LOG_FILE 2>&1

cp -r build/AlbanyFELIXInstall/bin/* $EXE_DIR
chmod -R 0755 $EXE_DIR

