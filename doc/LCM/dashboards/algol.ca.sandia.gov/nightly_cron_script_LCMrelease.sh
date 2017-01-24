#!/bin/sh

TESTDIR=/home/lcm/LCM/nightlyCDash 

cd $TESTDIR 

rm -rf repos
rm -rf build
rm -rf ctest_nightly.cmake
rm -rf nightly_log*
rm -rf results*

export PATH=$PATH:/usr/lib64/openmpi/bin

now=$(date +"%m_%d_%Y-%H_%M")

#build Trilinos
cat trilinosLCMrelease ctest_nightly.cmake.frag >& ctest_nightly.cmake 
LOG_FILE=nightly_log_trilinos_LCM_release.txt
eval "env  TEST_DIRECTORY=$TESTDIR SCRIPT_DIRECTORY=$TESTDIR ctest -VV -S $TESTDIR/ctest_nightly.cmake" > $TESTDIR/$LOG_FILE 2>&1

#build Albany and run tests
rm ctest_nightly.cmake 
cat albanyLCMrelease ctest_nightly.cmake.frag >& ctest_nightly.cmake  
LOG_FILE=nightly_log_albany_LCM_release.txt
eval "env  TEST_DIRECTORY=$TESTDIR SCRIPT_DIRECTORY=$TESTDIR ctest -VV -S $TESTDIR/ctest_nightly.cmake" > $TESTDIR/$LOG_FILE 2>&1

#process results of Albany ctest and send email 
bash process_results_albanyLCMrelease.sh 
bash send_email_LCM_release.sh 

