#!/bin/sh

cd /home/ikalash/nightlyAlbanyCDash

now=$(date +"%m_%d_%Y-%H_%M")

source mockba_modules.sh >& modules.out 
LOG_FILE=/home/ikalash/nightlyAlbanyCDash/nightly_log_cismAlbany.txt

#unset HTTPS_PROXY
#unset HTTP_PROXY
#export http_proxy=wwwproxy.ca.sandia.gov:80
#export https_proxy=wwwproxy.ca.sandia.gov:80

#env | grep -i proxy 

#the following is a hack...
cp mpi.mod /home/ikalash/nightlyAlbanyCDash/repos/cism-piscees/libglimmer

eval "env TEST_DIRECTORY=/home/ikalash/nightlyAlbanyCDash SCRIPT_DIRECTORY=/home/ikalash/nightlyAlbanyCDash ctest -VV -S /home/ikalash/nightlyAlbanyCDash/ctest_nightly_cismAlbany.cmake" > $LOG_FILE 2>&1
#eval "env https_proxy='https://wwwproxy.ca.sandia.gov:80' http_proxy='http://wwwproxy.ca.sandia.gov:80' HTTPS_PROXY='https://wwwproxy.ca.sandia.gov:80' HTTP_PROXY='http://wwwproxy.ca.sandia.gov' TEST_DIRECTORY=/home/ikalash/nightlyAlbanyCDash SCRIPT_DIRECTORY=/home/ikalash/nightlyAlbanyCDash ctest -VV -S /home/ikalash/nightlyAlbanyCDash/ctest_nightly_albany.cmake" > $LOG_FILE 2>&1
