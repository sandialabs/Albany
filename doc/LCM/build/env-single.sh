#!/bin/bash

SCRIPT_NAME=`basename $0`
PACKAGE=$1
TOOL_CHAIN=$2
BUILD_TYPE=$3
NUM_PROCS=$4
LCM_DIR=`pwd`
TRILINOS="trilinos"


if [ -z "$PACKAGE" ]; then
    echo "Specifiy package [trilinos|albany]"
    exit 1
fi

if [ -z "$TOOL_CHAIN" ]; then
    echo "Specify tool chain [gcc|clang]"
    exit 1
fi

if [ -z "$BUILD_TYPE" ]; then
    echo "Specify build type [debug|release]"
    exit 1
fi

if [ -z "$NUM_PROCS" ]; then
    NUM_PROCS="1"
fi

case "$PACKAGE" in
    trilinos)
	PACKAGE_STRING="TRILINOS"
	PACKAGE_NAME="Trilinos"
	;;
    albany)
	PACKAGE_STRING="ALBANY"
	PACKAGE_NAME="Albany"
	;;
    *)
	echo "Unrecognized package option"
	exit 1
	;;
esac

case "$TOOL_CHAIN" in
    gcc)
	export OMPI_CC=`which gcc`
	export OMPI_CXX=`which g++`
	export OMPI_FC=`which gfortran`
	CMAKE_CXX_FLAGS="-ansi -Wall -pedantic -Wno-long-long"
	;;
    clang)
	export OMPI_CC=`which clang`
	export OMPI_CXX=`which clang++`
	export OMPI_FC=`which gfortran`
	CMAKE_CXX_FLAGS="-Weverything -pedantic -Wno-long-long -Wno-documentation"
	;;
    *)
	echo "Unrecognized tool chain option"
	exit 1
	;;
esac

case "$BUILD_TYPE" in
    debug)
	BUILD_STRING="DEBUG"
	;;
    release)
	BUILD_STRING="RELEASE"
	;;
    *)
	echo "Unrecognized build type option"
	exit 1
	;;
esac

CONFIG_FILE="$PACKAGE-config.sh"
BUILD=$TOOL_CHAIN-$BUILD_TYPE
PACKAGE_DIR="$LCM_DIR/$PACKAGE_NAME"
# Install directory for trilinos only
INSTALL_DIR="$LCM_DIR/$TRILINOS-install-$BUILD"
BUILD_DIR="$LCM_DIR/$PACKAGE-build-$BUILD"
PREFIX="$PACKAGE-$TOOL_CHAIN-$BUILD_TYPE"
BUILD_LOG="$LCM_DIR/$PREFIX-build.log"
ERROR_LOG="$LCM_DIR/$PREFIX-error.log"
TEST_LOG="$LCM_DIR/$PREFIX-test.log"
HOST=`hostname`
FROM="name@address"
TO="name@address"
