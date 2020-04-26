#!/bin/bash

NUM_PROCS=`nproc`
export TEST_DIR=`pwd`
export MODULEPATH=$TEST_DIR/Albany/doc/LCM/modulefiles:$MODULEPATH

# trilinos required before albany
PACKAGES="trilinos albany"
ARCHES="serial"
TOOL_CHAINS="intel"
BUILD_TYPES="release"
