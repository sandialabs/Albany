#!/bin/bash

if [ "$1" == "" ]; then
  echo "Argument 1 (case num) not provided!" 
  exit 0
fi

build=$1

if [ $build = "regular" ]; then
  name="albany"
fi
if [ $build = "cali" ]; then
  name="cism-albany"
fi

sed -e "/CDASH-ALBANY-FILE.TXT/r /project/projectdirs/piscees/nightlyCoriCDash/cdash-$name-frag.txt" -e "/CDASH-ALBANY-FILE.TXT/d" "/project/projectdirs/piscees/nightlyCoriCDash/ctest_nightly_"$name"_build_tmp.cmake" >& /project/projectdirs/piscees/nightlyCoriCDash/ctest_nightly_"$name"_build.cmake
