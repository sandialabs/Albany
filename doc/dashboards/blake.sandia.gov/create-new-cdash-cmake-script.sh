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
buildtype=$2

if [ $compiler = "gcc" ]; then
if [ $buildtype = "release" ]; then
  sed -e '/CDASH-TRILINOS-GCC-RELEASE-FILE.TXT/r /home/projects/albany/nightlyCDashTrilinosBlake/cdash-gcc-release-frag.txt' -e '/CDASH-TRILINOS-GCC-RELEASE-FILE.TXT/d' /home/projects/albany/nightlyCDashTrilinosBlake/ctest_nightly_trilinos_gcc_release_tmp.cmake >& /home/projects/albany/nightlyCDashTrilinosBlake/ctest_nightly_trilinos_gcc_release.cmake
fi
if [ $buildtype = "debug" ]; then
  sed -e '/CDASH-TRILINOS-GCC-DEBUG-FILE.TXT/r /home/projects/albany/nightlyCDashTrilinosBlake/cdash-gcc-debug-frag.txt' -e '/CDASH-TRILINOS-GCC-DEBUG-FILE.TXT/d' /home/projects/albany/nightlyCDashTrilinosBlake/ctest_nightly_trilinos_gcc_debug_tmp.cmake >& /home/projects/albany/nightlyCDashTrilinosBlake/ctest_nightly_trilinos_gcc_debug.cmake
fi
fi
