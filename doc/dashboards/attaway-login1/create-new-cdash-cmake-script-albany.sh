#!/bin/bash

if [ "$1" == "" ]; then
  echo "Argument 1 (case num) not provided!" 
  exit 0
fi

build=$1

if [ $build = "regular" ]; then
  name="albany"
fi
if [ $build = "no-epetra" ]; then
  name="albanyT"
fi
if [ $build = "fpe" ]; then
  name="albanyFPE"
fi
if [ $build = "openmp" ]; then
  name="albanyOpenmp"
fi
if [ $build = "cali" ]; then
  name="cismAlbany"
fi

sed -e "/CDASH-ALBANY-FILE.TXT/r /projects/albany/nightlyCDash/cdash-$name-frag.txt" -e "/CDASH-ALBANY-FILE.TXT/d" "/projects/albany/nightlyCDash/ctest_nightly_"$name"_intel_serial_build_tmp.cmake" >& /projects/albany/nightlyCDash/ctest_nightly_"$name"_intel_serial_build.cmake
