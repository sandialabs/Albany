#!/bin/bash -ex

cd "$LCM_DIR"

PACKAGE=$1
THREADS=$2
VERBOSITY=$3
BUILD_ID_STRING=$ARCH-$TOOL_CHAIN-$BUILD_TYPE
PREFIX=$PACKAGE-$BUILD_ID_STRING
LOG_FILE="$LCM_DIR/${PREFIX}.log"

case "$THREADS" in
    -V)
        VERBOSITY="-V"
        THREADS=""
    ;;
    -VV)
        VERBOSITY="-VV"
        THREADS=""
    ;;
    *)
    ;;
esac

if [ -z $VERBOSITY ]; then
    VERBOSITY="-VV"
fi

ctest $VERBOSITY --timeout 180 -S $LCM_DIR/Albany/doc/LCM/build/lcm_build.cmake \
-DSCRIPT_NAME:STRING=`basename $0` \
-DPACKAGE:STRING=$PACKAGE \
-DBUILD_THREADS:STRING=$THREADS \
| tee $LOG_FILE
