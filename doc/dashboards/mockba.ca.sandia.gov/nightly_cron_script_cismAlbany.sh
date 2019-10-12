#!/bin/sh

cd /home/ikalash/nightlyAlbanyCDash

cat cismAlbany ctest_nightly.cmake.frag >& ctest_nightly.cmake  

#the following is a hack...
cp mpi.mod /home/ikalash/nightlyAlbanyCDash/repos/cism-piscees/libglimmer

now=$(date +"%m_%d_%Y-%H_%M")
source mockba_modules.sh >& modules_cali.out 
LOG_FILE=/home/ikalash/nightlyAlbanyCDash/nightly_log_cismAlbany.txt

eval "env  TEST_DIRECTORY=/home/ikalash/nightlyAlbanyCDash SCRIPT_DIRECTORY=/home/ikalash/nightlyAlbanyCDash ctest -VV -S /home/ikalash/nightlyAlbanyCDash/ctest_nightly.cmake" > $LOG_FILE 2>&1

