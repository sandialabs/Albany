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

source convert-cmake-to-cdash.sh
source create-new-cdash-cmake-script.sh

LOG_FILE=$BASE_DIR/nightly_log_trilinos_download.txt
BUILD_OPTION="download"
eval "env BUILD_OPTION=$BUILD_OPTION TEST_DIRECTORY=$BASE_DIR SCRIPT_DIRECTORY=$BASE_DIR ctest -VV -S $BASE_DIR/ctest_nightly_trilinos.cmake" > $LOG_FILE 2>&1

rm -rf repos/Trilinos/packages/kokkos
cp -r /nightlyAlbanyTests/kokkos-4-extended-types/ repos/Trilinos/packages/kokkos

LOG_FILE=$BASE_DIR/nightly_log_trilinos_build.txt
BUILD_OPTION="build"
eval "env BUILD_OPTION=$BUILD_OPTION TEST_DIRECTORY=$BASE_DIR SCRIPT_DIRECTORY=$BASE_DIR ctest -VV -S $BASE_DIR/ctest_nightly_trilinos.cmake" > $LOG_FILE 2>&1

