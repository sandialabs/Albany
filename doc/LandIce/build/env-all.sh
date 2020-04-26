#!/bin/bash

NUM_PROCS=`nproc`
export LCM_DIR=`pwd`
export MODULEPATH=$LCM_DIR/Albany/doc/LCM/modulefiles
unset https_proxy
unset http_proxy

# trilinos required before albany
PACKAGES="trilinos albany"
ARCHES="serial"
TOOL_CHAINS="gcc"
BUILD_TYPES="release"
