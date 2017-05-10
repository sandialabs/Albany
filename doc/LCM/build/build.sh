#!/bin/bash -ex

cd "$LCM_DIR"

ctest -VV -S $LCM_DIR/Albany/doc/LCM/build/lcm_build.cmake \
-DSCRIPT_NAME:STRING=`basename $0` \
-DPACKAGE:STRING=$1 \
-DBUILD_THREADS:STRING=$2
