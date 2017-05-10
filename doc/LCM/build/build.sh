#!/bin/bash -ex

cd "$LCM_DIR"

cmake -P $LCM_DIR/Albany/doc/LCM/build/lcm_build.cmake \
-DSCRIPT_NAME=`basename $0` \
-DPACKAGE=$1 \
-DBUILD_THREADS=$2
