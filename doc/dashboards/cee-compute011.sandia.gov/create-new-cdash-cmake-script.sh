#!/bin/bash

if [ "$1" == "" ]; then
  echo "Argument 1 (case num) not provided!" 
  exit 0
fi

compiler=$1

if [ $compiler = "intel" ]; then
  sed -e '/CDASH-TRILINOS-INTEL-FILE.TXT/r /projects/albany/nightlyAlbanyCDash/cdash-intel-frag.txt' -e '/CDASH-TRILINOS-INTEL-FILE.TXT/d' /projects/albany/nightlyAlbanyCDash/ctest_nightly_tmp.cmake >& /projects/albany/nightlyAlbanyCDash/ctest_nightly.cmake
fi
if [ $compiler = "clang" ]; then
  sed -e '/CDASH-TRILINOS-CLANG-FILE.TXT/r /projects/albany/nightlyAlbanyCDash/cdash-clang-frag.txt' -e '/CDASH-TRILINOS-CLANG-FILE.TXT/d' /projects/albany/nightlyAlbanyCDash/ctest_nightly.cmake >& /projects/albany/nightlyAlbanyCDash/tmp.cmake
  mv /projects/albany/nightlyAlbanyCDash/tmp.cmake /projects/albany/nightlyAlbanyCDash/ctest_nightly.cmake 
fi
if [ $compiler = "gcc" ]; then
  sed -e '/CDASH-TRILINOS-GCC-FILE.TXT/r /projects/albany/nightlyAlbanyCDash/cdash-gcc-frag.txt' -e '/CDASH-TRILINOS-GCC-FILE.TXT/d' /projects/albany/nightlyAlbanyCDash/ctest_nightly.cmake >& /projects/albany/nightlyAlbanyCDash/tmp.cmake
  mv /projects/albany/nightlyAlbanyCDash/tmp.cmake /projects/albany/nightlyAlbanyCDash/ctest_nightly.cmake 
fi
