#!/bin/bash

# WARNING: This file is generated automatically. Any changes made here
# will be lost when the package is configured again.  Any permanent
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
 -D ENABLE_LANDICE:BOOL=ON \
 -D ENABLE_UNIT_TESTS:BOOL=ON \
 -D ENABLE_CHECK_FPE:BOOL=lcm_fpe_switch \
 -D ENABLE_FLUSH_DENORMALS:BOOL=lcm_denormal_switch \
 -D ALBANY_ENABLE_FORTRAN:BOOL=OFF \
 -D ENABLE_SLFAD:BOOL=lcm_enable_slfad \
 lcm_slfad_size \
 lcm_package_dir
