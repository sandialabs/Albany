#!/bin/bash

#if [ "$1" == "" ]; then
#  echo "Argument 1 (case num) not provided!" 
#  exit 0
#fi
#if [ "$2" == "" ]; then
#  echo "Argument 2 (case num) not provided!" 
#  exit 0
#fi

#compiler=$1
#kokkosnode=$2

#if [ $compiler = "intel" ]; then
#if [ $kokkosnode = "serial" ]; then
  sed -e '/CDASH-TRILINOS-RELEASE-FILE.TXT/r /home/ikalash/nightlyAlbanyCDash/cdash-hayka-frag.txt' -e '/CDASH-TRILINOS-RELEASE-FILE.TXT/d' /home/ikalash/nightlyAlbanyCDash/ctest_nightly_trilinos_serial_tmp.cmake >& /home/ikalash/nightlyAlbanyCDash/ctest_nightly_trilinos_serial.cmake
#fi
#fi
