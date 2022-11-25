#!/bin/bash

if [ "$1" == "" ]; then
  echo "Argument 1 (case num) not provided!" 
  exit 0
fi
if [ "$2" == "" ]; then
  echo "Argument 2 (case num) not provided!" 
  exit 0
fi

compiler=$1
kokkosnode=$2

if [ $compiler = "intel" ]; then
if [ $kokkosnode = "serial" ]; then
  sed -e '/CDASH-TRILINOS-INTEL-SERIAL-FILE.TXT/r /home/ikalash/albany/nightlyCDash/cdash-intel-serial-frag.txt' -e '/CDASH-TRILINOS-INTEL-SERIAL-FILE.TXT/d' /home/ikalash/albany/nightlyCDash/ctest_nightly_trilinos_intel_serial_tmp.cmake >& /home/ikalash/albany/nightlyCDash/ctest_nightly_trilinos_intel_serial.cmake
fi
fi
