#!/bin/bash

if [ "$1" == "" ]; then
  echo "Argument 1 (case num) not provided!" 
  exit 0
fi

kokkosnode=$1

if [ $kokkosnode = "serial" ]; then
  sed -e '/CDASH-TRILINOS-INTEL-SERIAL-FILE.TXT/r /home/projects/albany/nightlyCDashTrilinosBlake/cdash-serial-frag.txt' -e '/CDASH-TRILINOS-INTEL-SERIAL-FILE.TXT/d' /home/projects/albany/nightlyCDashTrilinosBlake/ctest_nightly_trilinos_serial_tmp.cmake >& /home/projects/albany/nightlyCDashTrilinosBlake/ctest_nightly_trilinos_serial.cmake
fi
if [ $kokkosnode = "openmp" ]; then
  sed -e '/CDASH-TRILINOS-INTEL-OPENMP-FILE.TXT/r /home/projects/albany/nightlyCDashTrilinosBlake/cdash-openmp-frag.txt' -e '/CDASH-TRILINOS-INTEL-OPENMP-FILE.TXT/d' /home/projects/albany/nightlyCDashTrilinosBlake/ctest_nightly_trilinos_openmp_tmp.cmake >& /home/projects/albany/nightlyCDashTrilinosBlake/ctest_nightly_trilinos_openmp.cmake
fi
