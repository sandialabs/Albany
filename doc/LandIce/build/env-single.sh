#!/bin/bash

SCRIPT_NAME=`basename $0`
PACKAGE=$1
NUM_PROCS=$2
export TEST_DIR=`pwd`
TRILINOS="trilinos"
INTEL_DIR=/opt/intel

# Some basic error checking.
if [ -z "$PACKAGE" ]; then
    echo "Specify package [trilinos|albany]"
    exit 1
fi

if [ -z "$ARCH" ]; then
    echo "Specify architecture [serial|openmp|pthreads|cuda]"
    exit 1
fi

if [ -z "$TOOL_CHAIN" ]; then
    echo "Specify tool chain [gcc|clang|intel|pgi]"
    exit 1
fi

if [ -z "$BUILD_TYPE" ]; then
    echo "Specify build type [debug|release|profile|small|mixed]"
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
	echo "Unrecognized package option in env-single: $PACKAGE"
	exit 1
	;;
esac

case "$ARCH" in
    serial)
	ARCH_STRING="SERIAL"
	ARCH_NAME="Serial"
	;;
    openmp)
	ARCH_STRING="OPENMP"
	ARCH_NAME="Open MP"
	;;
    pthreads)
	ARCH_STRING="PTHREADS"
	ARCH_NAME="POSIX Threads"
	;;
    cuda)
	ARCH_STRING="CUDA"
	ARCH_NAME="Cuda"
	;;
    *)
	echo "Unrecognized architecture option in env-single: $ARCH"
	exit 1
	;;
esac

NVCC_WRAPPER="$TEST_DIR/Trilinos/packages/kokkos/config/nvcc_wrapper"

case "$TOOL_CHAIN" in
    gcc)
	if [ -z ${CC+x} ]; then CC=`which gcc`; fi
	case "$ARCH" in
	    serial)
		if [ -z ${CXX+x} ]; then CXX=`which g++`; fi
		;;
	    openmp)
		if [ -z ${CXX+x} ]; then CXX=`which g++`; fi
		;;
	    pthreads)
		if [ -z ${CXX+x} ]; then CXX=`which g++`; fi
		;;
	    cuda)
		if [ -z ${CXX+x} ]; then
		    CXX="$NVCC_WRAPPER";
		else
		    export NVCC_WRAPPER_DEFAULT_COMPILER="$CXX";
		    CXX="$NVCC_WRAPPER";
		fi
		;;
	    *)
		echo "Unrecognized architecture option in env-single: $ARCH"
		exit 1
		;;
	esac
	if [ -z ${FC+x} ]; then FC=`which gfortran`; fi
	;;
    clang)
	if [ -z ${CC+x} ]; then CC=`which clang`; fi
	if [ -z ${CXX+x} ]; then CXX=`which clang++`; fi
	if [ -z ${FC+x} ]; then FC=`which gfortran`; fi
	;;
    intel)
	source $INTEL_DIR/bin/compilervars.sh intel64
	if [ -z ${CC+x} ]; then CC=`which icc`; fi
	if [ -z ${CXX+x} ]; then CXX=`which icpc`; fi
	if [ -z ${FC+x} ]; then FC=`which ifort`; fi
	;;
    pgi)
	if [ -z ${CC+x} ]; then CC=`which pgcc`; fi
	if [ -z ${CXX+x} ]; then CXX=`which pgc++`; fi
	if [ -z ${FC+x} ]; then FC=`which pgfortran`; fi
	;;
    *)
	echo "Unrecognized tool chain option in env-single: $TOOL_CHAIN"
	exit 1
	;;
esac
export OMPI_CC="$CC"
export OMPI_CXX="$CXX"
export OMPI_FC="$FC"

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
	    pgi)
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
	    pgi)
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
	    pgi)
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
	    pgi)
		;;
	    *)
		;;
	esac
	;;
    mixed)
	BUILD_STRING="RELEASE"
	case "$TOOL_CHAIN" in
	    gcc)
		;;
	    clang)
		;;
	    intel)
		;;
	    pgi)
		;;
	    *)
		;;
	esac
	;;
    *)
	echo "Unrecognized build type option in env-single: $BUILD_TYPE"
	exit 1
	;;
esac

# Setup flags with the info gathered above.
CONFIG_FILE="$PACKAGE-config.sh"
DTK_FRAG="dtk-frag.sh"
BUILD=$ARCH-$TOOL_CHAIN-$BUILD_TYPE
PACKAGE_DIR="$TEST_DIR/$PACKAGE_NAME"
# Install directory for trilinos only
INSTALL_DIR="$TEST_DIR/$TRILINOS-install-$BUILD"
BUILD_DIR="$TEST_DIR/$PACKAGE-build-$BUILD"
PREFIX="$PACKAGE-$BUILD"
BUILD_LOG="$TEST_DIR/$PREFIX-build.log"
ERROR_LOG="$TEST_DIR/$PREFIX-error.log"
STATUS_LOG="$TEST_DIR/$PREFIX-status.log"
TEST_LOG="$TEST_DIR/$PREFIX-test.log"
HOST=`hostname`
FROM=`whoami`"@sandia.gov"
TO="albany-regression@software.sandia.gov"
PROJECT_XML_FILE="Project.xml"
CTEST_FILE="ctest.cmake"
CTEST_TYPE="Nightly"
