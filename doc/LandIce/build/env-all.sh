#!/bin/bash

NUM_PROCS=`nproc`
export TEST_DIR=`pwd`
export MODULEPATH=$TEST_DIR/Albany/doc/LandIce/modulefiles
unset https_proxy
unset http_proxy

# trilinos required before albany
PACKAGES="trilinos albany"
ARCHES="serial"
TOOL_CHAINS="gcc"
BUILD_TYPES="release"
