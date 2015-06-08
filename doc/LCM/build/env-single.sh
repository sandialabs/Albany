#!/bin/bash

SCRIPT_NAME=`basename $0`
PACKAGE=$1
TOOL_CHAIN=$2
BUILD_TYPE=$3
NUM_PROCS=$4
LCM_DIR=`pwd`
TRILINOS="trilinos"
INTEL_DIR=/opt/intel

# Some basic error checking.
if [ -z "$PACKAGE" ]; then
    echo "Specifiy package [trilinos|albany]"
    exit 1
fi

if [ -z "$TOOL_CHAIN" ]; then
    echo "Specify tool chain [gcc|clang|intel]"
    exit 1
fi

if [ -z "$BUILD_TYPE" ]; then
    echo "Specify build type [debug|release|profile|small]"
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
	;;
    clang)
	export OMPI_CC=`which clang`
	export OMPI_CXX=`which clang++`
	export OMPI_FC=`which gfortran`
	;;
    intel)
	source $INTEL_DIR/bin/compilervars.sh intel64
	export OMPI_CC=`which icc`
	export OMPI_CXX=`which icpc`
	export OMPI_FC=`which ifort`
	;;
    *)
	echo "Unrecognized tool chain option"
	exit 1
	;;
esac

# This is here to add or change compiler flags in addition to those
# specified by CMake during configuration. Right now they are empty
# (at least for gcc) as any additional modifications here seem to
# interfere with the NVidia CUDA compiler.
case "$BUILD_TYPE" in
    debug)
	BUILD_STRING="DEBUG"
	case "$TOOL_CHAIN" in
	    gcc)
		;;
	    clang)
		;;
	    intel)
		;;
	    *)
		;;
	esac
	;;
    release)
	BUILD_STRING="RELEASE"
	case "$TOOL_CHAIN" in
	    gcc)
		;;
	    clang)
		;;
	    intel)
		;;
	    *)
		;;
	esac
	;;
    profile)
	BUILD_STRING="RELWITHDEBINFO"
	case "$TOOL_CHAIN" in
	    gcc)
		;;
	    clang)
		;;
	    intel)
		;;
	    *)
		;;
	esac
	;;
    small)
	BUILD_STRING="MINSIZEREL"
	case "$TOOL_CHAIN" in
	    gcc)
		;;
	    clang)
		;;
	    intel)
		;;
	    *)
		;;
	esac
	;;
    *)
	echo "Unrecognized build type option"
	exit 1
	;;
esac

# Setup flags with the info gathered above.
CONFIG_FILE="$PACKAGE-config.sh"
BUILD=$TOOL_CHAIN-$BUILD_TYPE
PACKAGE_DIR="$LCM_DIR/$PACKAGE_NAME"
# Install directory for trilinos only
INSTALL_DIR="$LCM_DIR/$TRILINOS-install-$BUILD"
BUILD_DIR="$LCM_DIR/$PACKAGE-build-$BUILD"
PREFIX="$PACKAGE-$TOOL_CHAIN-$BUILD_TYPE"
BUILD_LOG="$LCM_DIR/$PREFIX-build.log"
ERROR_LOG="$LCM_DIR/$PREFIX-error.log"
STATUS_LOG="$LCM_DIR/$PREFIX-status.log"
TEST_LOG="$LCM_DIR/$PREFIX-test.log"
HOST=`hostname`
FROM="amota@sandia.gov"
TO="albany-regression@software.sandia.gov"
# Set directory flags so that the appropriate shared objects and executables
# can be found.
PATH="$INSTALL_DIR/bin:$PATH"
LD_LIBRARY_PATH="$INSTALL_DIR/lib:$LD_LIBRARY_PATH"
