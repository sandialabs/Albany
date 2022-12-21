#!/bin/bash

if [ "$1" == "" ]; then
  echo "Argument 1 (case num) not provided!" 
  exit 0
fi

build=$1

if [ $build = "intel-serial" ]; then
  name="albany-intel-serial"
fi
if [ $build = "intel-openmp" ]; then
  name="albany-intel-openmp"
fi
if [ $build = "gcc-serial" ]; then
  name="albany-gcc-serial"
fi
if [ $build = "sfad" ]; then
  name="albany-sfad"
fi

sed -e "/CDASH-ALBANY-FILE.TXT/r /home/projects/albany/nightlyCDashAlbanyBlake/cdash-$name-frag.txt" -e "/CDASH-ALBANY-FILE.TXT/d" "/home/projects/albany/nightlyCDashAlbanyBlake/ctest_nightly_"$name"_tmp.cmake" >& /home/projects/albany/nightlyCDashAlbanyBlake/ctest_nightly_$name.cmake
