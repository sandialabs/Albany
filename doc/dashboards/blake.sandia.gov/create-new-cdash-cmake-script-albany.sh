#!/bin/bash

if [ "$1" == "" ]; then
  echo "Argument 1 (case num) not provided!" 
  exit 0
fi

build=$1

if [ $build = "gcc-release" ]; then
  name="albany-gcc-release"
fi
if [ $build = "gcc-debug" ]; then
  name="albany-gcc-debug"
fi
if [ $build = "intel-release" ]; then
  name="albany-intel-release"
fi
if [ $build = "sfad" ]; then
  name="albany-sfad"
fi

sed -e "/CDASH-ALBANY-FILE.TXT/r /home/projects/albany/nightlyCDashAlbanyBlake/cdash-$name-frag.txt" -e "/CDASH-ALBANY-FILE.TXT/d" "/home/projects/albany/nightlyCDashAlbanyBlake/ctest_nightly_"$name"_tmp.cmake" >& /home/projects/albany/nightlyCDashAlbanyBlake/ctest_nightly_$name.cmake
