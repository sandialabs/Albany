#!/bin/bash

if [ "$1" == "" ]; then
  echo "Argument 1 (case num) not provided!" 
  exit 0
fi

build=$1

if [ $build = "regular" ]; then
  name="albany"
fi

sed -e "/CDASH-ALBANY-FILE.TXT/r /projects/albany/nightlyAlbanyCDash/cdash-$name-frag.txt" -e "/CDASH-ALBANY-FILE.TXT/d" "/projects/albany/nightlyAlbanyCDash/ctest_nightly_tmp.cmake" >& /projects/albany/nightlyAlbanyCDash/ctest_nightly.cmake
