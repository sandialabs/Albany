#!/bin/csh

BASE_DIR=/home/projects/albany/nightlyCDashAlbanyBlake
cd $BASE_DIR

rm -rf build-intel
rm -rf repos-intel
rm -rf nightly*Intel*.txt
rm -rf nightly*SFad*.txt
rm -rf intel_modules.out 

unset http_proxy
unset https_proxy

export INTEL_LICENSE_FILE=/home/projects/x86-64/intel/licenses/USE_SERVER-ohpc.lic
source blake_intel_modules.sh >& intel_modules.out  

export OMP_NUM_THREADS=1

LOG_FILE=$BASE_DIR/nightly_log_blakeAlbanyIntelSerial.txt

bash convert-cmake-to-cdash-albany.sh intel-serial
bash create-new-cdash-cmake-script-albany.sh intel-serial

eval "env  TEST_DIRECTORY=$BASE_DIR SCRIPT_DIRECTORY=$BASE_DIR ctest -VV -S $BASE_DIR/ctest_nightly_albany-intel-serial.cmake" > $LOG_FILE 2>&1

