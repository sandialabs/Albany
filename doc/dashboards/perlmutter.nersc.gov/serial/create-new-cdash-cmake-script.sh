#!/bin/bash

sed -e '/CDASH-TRILINOS-FILE.TXT/r /pscratch/sd/j/jwatkins/nightlyCDashPerlmutterSerial/cdash-frag.txt' -e '/CDASH-TRILINOS-FILE.TXT/d' /pscratch/sd/j/jwatkins/nightlyCDashPerlmutterSerial/ctest_nightly_trilinos_tmp.cmake >& /pscratch/sd/j/jwatkins/nightlyCDashPerlmutterSerial/ctest_nightly_trilinos.cmake
