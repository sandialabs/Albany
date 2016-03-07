#!/bin/bash

SCRIPT_NAME=`basename $0`
NUM_PROCS=`nproc`
export LCM_DIR=`pwd`
export MODULEPATH=$LCM_DIR/Albany/doc/LCM/modulefiles

# trilinos required before albany
PACKAGES="trilinos albany"
ARCHES="serial"
BUILD_TYPES="release"
TOOL_CHAINS="gcc"
