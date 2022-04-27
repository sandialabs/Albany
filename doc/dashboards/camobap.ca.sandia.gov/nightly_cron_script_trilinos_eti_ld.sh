#!/bin/csh

BASE_DIR=/nightlyCDash
cd $BASE_DIR

#rm -rf repos
#rm -rf build
#rm -rf nightly_log*
#rm -rf results*
#rm -rf modules*out 
#source clean-up.sh 

export https_proxy="http://proxy.ca.sandia.gov:80"
export http_proxy="http://proxy.ca.sandia.gov:80"
export PATH=$PATH:/tpls/install/ninja/build-cmake

source convert-cmake-to-cdash-longdouble-eti.sh
source create-new-cdash-cmake-script-eti-ld.sh

LOG_FILE=$BASE_DIR/nightly_log_trilinos_build_eti_ld.txt
BUILD_OPTION="build"
eval "env BUILD_OPTION=$BUILD_OPTION TEST_DIRECTORY=$BASE_DIR SCRIPT_DIRECTORY=$BASE_DIR ctest -VV -S $BASE_DIR/ctest_nightly_trilinos_eti_ld.cmake" > $LOG_FILE 2>&1

