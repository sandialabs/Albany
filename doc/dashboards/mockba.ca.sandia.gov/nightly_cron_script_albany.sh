#!/bin/sh

cd /home/ikalash/nightlyAlbanyCDash

rm -rf repos
rm -rf build
rm -rf ctest_nightly.cmake.work
rm -rf nightly_log*
rm -rf results*
rm -rf *out 


now=$(date +"%m_%d_%Y-%H_%M")

source mockba_modules.sh >& modules.out 
LOG_FILE=/home/ikalash/nightlyAlbanyCDash/nightly_log.txt

#unset HTTPS_PROXY
#unset HTTP_PROXY
#export http_proxy=wwwproxy.ca.sandia.gov:80
#export https_proxy=wwwproxy.ca.sandia.gov:80

#env | grep -i proxy 

eval "env TEST_DIRECTORY=/home/ikalash/nightlyAlbanyCDash SCRIPT_DIRECTORY=/home/ikalash/nightlyAlbanyCDash ctest -VV -S /home/ikalash/nightlyAlbanyCDash/ctest_nightly_albany.cmake" > $LOG_FILE 2>&1
#eval "env https_proxy='https://wwwproxy.ca.sandia.gov:80' http_proxy='http://wwwproxy.ca.sandia.gov:80' HTTPS_PROXY='https://wwwproxy.ca.sandia.gov:80' HTTP_PROXY='http://wwwproxy.ca.sandia.gov' TEST_DIRECTORY=/home/ikalash/nightlyAlbanyCDash SCRIPT_DIRECTORY=/home/ikalash/nightlyAlbanyCDash ctest -VV -S /home/ikalash/nightlyAlbanyCDash/ctest_nightly_albany.cmake" > $LOG_FILE 2>&1
