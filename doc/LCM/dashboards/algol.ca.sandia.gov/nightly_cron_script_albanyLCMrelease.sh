#!/bin/sh

TESTDIR=/home/ikalash/LCM/nightlyCDash 

cd $TESTDIR 

rm -rf repos
rm -rf build
rm -rf ctest_nightly.cmake
rm -rf nightly_log*
rm -rf results*


cat albanyLCMrelease ctest_nightly.cmake.frag >& ctest_nightly.cmake  

export PATH=$PATH:/usr/lib64/openmpi/bin


now=$(date +"%m_%d_%Y-%H_%M")
LOG_FILE=nightly_log_LCM_release.txt

eval "env  TEST_DIRECTORY=$TESTDIR SCRIPT_DIRECTORY=$TESTDIR ctest -VV -S $TESTDIR/ctest_nightly.cmake" > $TESTDIR/$LOG_FILE 2>&1
bash process_results_albanyLCMrelease.sh 
bash send_email_LCM_release.sh 

