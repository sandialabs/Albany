#!/bin/sh

cd /home/ikalash/nightlyAlbanyCDash

source mockba_modules.sh >& modules.out 
LOG_FILE=/home/ikalash/nightlyAlbanyCDash/nightly_log_cismAlbany.txt

#unset HTTPS_PROXY
#unset HTTP_PROXY
#export http_proxy=wwwproxy.ca.sandia.gov:80
#export https_proxy=wwwproxy.ca.sandia.gov:80

#env | grep -i proxy 

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib64:/usr/include/boost:/home/ikalash/Downloads/libiconv-1.14/lib/.libs
export PATH=$PATH:/usr/lib64/openmpi/bin:/home/ikalash/Install/ParaView-4.4.0-Qt4-Linux-64bit/bin:/ascldap/users/ikalash/Trilinos/seacas-build/install/bin:/home/ikalash/Install/Cubit:/ascldap/users/ikalash/Install/MATLAB/R2018b/bin:/ascldap/users/ikalash/Trilinos/seacas-build/install/bin


eval "env TEST_DIRECTORY=/home/ikalash/nightlyAlbanyCDash SCRIPT_DIRECTORY=/home/ikalash/nightlyAlbanyCDash ctest -VV -S /home/ikalash/nightlyAlbanyCDash/ctest_nightly_cismAlbany.cmake" > $LOG_FILE 2>&1
#eval "env https_proxy='https://wwwproxy.ca.sandia.gov:80' http_proxy='http://wwwproxy.ca.sandia.gov:80' HTTPS_PROXY='https://wwwproxy.ca.sandia.gov:80' HTTP_PROXY='http://wwwproxy.ca.sandia.gov' TEST_DIRECTORY=/home/ikalash/nightlyAlbanyCDash SCRIPT_DIRECTORY=/home/ikalash/nightlyAlbanyCDash ctest -VV -S /home/ikalash/nightlyAlbanyCDash/ctest_nightly_albany.cmake" > $LOG_FILE 2>&1
