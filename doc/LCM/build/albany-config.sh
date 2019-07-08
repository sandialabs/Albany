#!/bin/bash

# WARNING: This file is generated automatically. Any changes made here
# will be lost when the package is configured again.  Any permament
# changes should go into the corresponding template at the top level
# LCM directory.

if [ -f ./CMakeCache.txt ]; then
    rm ./CMakeCache.txt
fi

if [ -d ./CMakeFiles ]; then
    rm ./CMakeFiles -rf
fi

# The Trilinos Dir is the same as the PREFIX entry from the
# Trilinos configuration script

cmake \
 -D ALBANY_CTEST_TIMEOUT:INTEGER=60 \
 -D ALBANY_TRILINOS_DIR:FILEPATH=lcm_install_dir \
 -D CMAKE_CXX_FLAGS:STRING="lcm_cxx_flags" \
 -D CMAKE_VERBOSE_MAKEFILE:BOOL=OFF \
 -D ENABLE_LCM:BOOL=ON \
 -D ENABLE_ATO:BOOL=OFF \
 -D ENABLE_LANDICE:BOOL=OFF \
 -D ENABLE_LAME:BOOL=OFF \
 -D ENABLE_LAMENT:BOOL=OFF \
 -D ENABLE_CHECK_FPE:BOOL=lcm_fpe_switch \
 -D ENABLE_FLUSH_DENORMALS:BOOL=lcm_denormal_switch \
 -D ALBANY_ENABLE_FORTRAN:BOOL=OFF \
 lcm_package_dir
