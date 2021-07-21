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
  sed -e '/CDASH-TRILINOS-INTEL-SERIAL-FILE.TXT/r /home/projects/albany/nightlyCDashTrilinosBlake/cdash-intel-serial-frag.txt' -e '/CDASH-TRILINOS-INTEL-SERIAL-FILE.TXT/d' /home/projects/albany/nightlyCDashTrilinosBlake/ctest_nightly_trilinos_intel_serial_tmp.cmake >& /home/projects/albany/nightlyCDashTrilinosBlake/ctest_nightly_trilinos_intel_serial.cmake
fi
if [ $kokkosnode = "openmp" ]; then
  sed -e '/CDASH-TRILINOS-INTEL-OPENMP-FILE.TXT/r /home/projects/albany/nightlyCDashTrilinosBlake/cdash-intel-openmp-frag.txt' -e '/CDASH-TRILINOS-INTEL-OPENMP-FILE.TXT/d' /home/projects/albany/nightlyCDashTrilinosBlake/ctest_nightly_trilinos_intel_openmp_tmp.cmake >& /home/projects/albany/nightlyCDashTrilinosBlake/ctest_nightly_trilinos_intel_openmp.cmake
fi
fi

if [ $compiler = "gcc" ]; then
if [ $kokkosnode = "serial" ]; then
  sed -e '/CDASH-TRILINOS-GCC-SERIAL-FILE.TXT/r /home/projects/albany/nightlyCDashTrilinosBlake/cdash-gcc-serial-frag.txt' -e '/CDASH-TRILINOS-GCC-SERIAL-FILE.TXT/d' /home/projects/albany/nightlyCDashTrilinosBlake/ctest_nightly_trilinos_gcc_serial_tmp.cmake >& /home/projects/albany/nightlyCDashTrilinosBlake/ctest_nightly_trilinos_gcc_serial.cmake
fi
fi
