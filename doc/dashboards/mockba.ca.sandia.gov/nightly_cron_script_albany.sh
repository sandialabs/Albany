#!/bin/sh

cd /home/ikalash/nightlyAlbanyCDash

rm -rf repos
rm -rf build
rm -rf ctest_nightly.cmake.work
rm -rf nightly_log*
rm -rf results*
rm -rf *out 

source mockba_modules.sh >& modules.out 
LOG_FILE=/home/ikalash/nightlyAlbanyCDash/nightly_log.txt

unset HTTPS_PROXY
unset HTTP_PROXY
#export http_proxy=proxy.ca.sandia.gov:80
#export https_proxy=proxy.ca.sandia.gov:80

#env | grep -i proxy 

#Hack for Albany block disc unit tests + CISM 
alias python=/projects/sems/install/rhel7-x86_64/sems/compiler/python/2.7.9/bin/python
alias python3=/projects/sems/install/rhel7-x86_64/sems/compiler/python/2.7.9/bin/python

bash convert-cmake-to-cdash.sh regular
bash create-new-cdash-cmake-script.sh regular

eval "env TEST_DIRECTORY=/home/ikalash/nightlyAlbanyCDash SCRIPT_DIRECTORY=/home/ikalash/nightlyAlbanyCDash ctest -VV -S /home/ikalash/nightlyAlbanyCDash/ctest_nightly_albany.cmake" > $LOG_FILE 2>&1
