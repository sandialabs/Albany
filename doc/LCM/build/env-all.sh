#!/bin/bash

SCRIPT_NAME=`basename $0`
NUM_PROCS=`nproc`
LCM_DIR=`pwd`

# trilinos required before albany
PACKAGES="trilinos albany"
TOOL_CHAINS="gcc clang"
BUILD_TYPES="debug release"
