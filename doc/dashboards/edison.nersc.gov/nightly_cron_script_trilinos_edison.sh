#!/bin/csh

cd /project/projectdirs/piscees/nightlyEdisonCDash

source edison_modules.sh 

rm -rf /project/projectdirs/piscees/nightlyEdisonCDash/repos
rm -rf /project/projectdirs/piscees/nightlyEdisonCDash/build
rm -rf /project/projectdirs/piscees/nightlyEdisonCDash/ctest_nightly.cmake.work
rm -rf /project/projectdirs/piscees/nightlyEdisonCDash/nightly_log*
rm -rf /project/projectdirs/piscees/nightlyEdisonCDash/results*
rm -rf /project/projectdirs/piscees/nightlyEdisonCDash/test_summary.txt

cat trilinosFELIX ctest_nightly.cmake.frag >& ctest_nightly.cmake  

now=$(date +"%m_%d_%Y-%H_%M")
LOG_FILE=/project/projectdirs/piscees/nightlyEdisonCDash/nightly_log_edisonTrilinos.txt

eval "env  TEST_DIRECTORY=/project/projectdirs/piscees/nightlyEdisonCDash SCRIPT_DIRECTORY=/project/projectdirs/piscees/nightlyEdisonCDash ctest -VV -S /project/projectdirs/piscees/nightlyEdisonCDash/ctest_nightly.cmake" > $LOG_FILE 2>&1

