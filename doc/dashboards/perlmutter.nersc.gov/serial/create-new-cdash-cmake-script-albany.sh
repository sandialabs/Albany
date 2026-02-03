#!/bin/bash

FAD_CONFIGURATION=${1}
FAD_SIZE=${2}

BASE_DIR=/pscratch/sd/j/jwatkins/nightlyCDashPerlmutterSerial

if [ "$FAD_CONFIGURATION" = "slfad" ] ; then
  sed -e "/CDASH-ALBANY-FILE.TXT/r ${BASE_DIR}/cdash-albany-frag-slfad.txt" -e "/CDASH-ALBANY-FILE.TXT/d" "${BASE_DIR}/ctest_nightly_albany_tmp.cmake" >& ${BASE_DIR}/ctest_nightly_albany_slfad.cmake
fi
if [ "$FAD_CONFIGURATION" = "sfad" ] && [ "$FAD_SIZE" = "12" ]; then
  sed -e "/CDASH-ALBANY-FILE.TXT/r ${BASE_DIR}/cdash-albany-frag-sfad12.txt" -e "/CDASH-ALBANY-FILE.TXT/d" "${BASE_DIR}/ctest_nightly_albany_tmp.cmake" >& ${BASE_DIR}/ctest_nightly_albany_sfad12.cmake
fi
if [ "$FAD_CONFIGURATION" = "sfad" ] && [ "$FAD_SIZE" = "24" ]; then
  sed -e "/CDASH-ALBANY-FILE.TXT/r ${BASE_DIR}/cdash-albany-frag-sfad24.txt" -e "/CDASH-ALBANY-FILE.TXT/d" "${BASE_DIR}/ctest_nightly_albany_tmp.cmake" >& ${BASE_DIR}/ctest_nightly_albany_sfad24.cmake
fi
