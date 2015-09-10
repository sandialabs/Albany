#!/bin/csh

cd /project/projectdirs/piscees/nightlyEdisonCDash

source edison_modules.sh 

rm -rf /project/projectdirs/piscees/nightlyEdisonCDash/ctest_nightly.cmake.work

cat cismAlbanyFELIX ctest_nightly.cmake.frag >& ctest_nightly.cmake  


now=$(date +"%m_%d_%Y-%H_%M")
LOG_FILE=/project/projectdirs/piscees/nightlyEdisonCDash/nightly_log_edisonCismAlbany.txt

eval "env  TEST_DIRECTORY=/project/projectdirs/piscees/nightlyEdisonCDash SCRIPT_DIRECTORY=/project/projectdirs/piscees/nightlyEdisonCDash ctest -VV -S /project/projectdirs/piscees/nightlyEdisonCDash/ctest_nightly.cmake" > $LOG_FILE 2>&1

cp -r build/EdisonCismAlbany/cism_driver/cism_driver /project/projectdirs/piscees/nightlyEdisonCDashExe
chmod -R 0755 /project/projectdirs/piscees/nightlyEdisonCDashExe

bash process_output.sh 
bash send_email.sh  
