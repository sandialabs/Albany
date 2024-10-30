#!/bin/bash

if [ "$1" == "" ]; then
  echo "Argument 1 (case num) not provided!" 
  exit 0
fi

build=$1

if [ $build = "regular" ]; then
  name="albany"
fi
if [ $build = "sfad" ]; then
  name="albanySFAD"
fi

sed -e "/CDASH-ALBANY-FILE.TXT/r /home/projects/albany/nightlyCDashWeaver/cdash-$name-frag.txt" -e "/CDASH-ALBANY-FILE.TXT/d" "/home/projects/albany/nightlyCDashWeaver/ctest_nightly_"$name"_tmp.cmake" >& /home/projects/albany/nightlyCDashWeaver/ctest_nightly_$name.cmake
