#!/bin/bash

if [ -f ./CMakeCache.txt ]; then
    rm CMakeCache.txt
fi

# The Trilinos Dir is the same as the PREFIX entry from the
# Trilinos configuration script

cmake \
 -D CMAKE_CXX_FLAGS:STRING="cmake_cxx_flags" \
 -D CMAKE_BUILD_TYPE:STRING="build_type" \
 -D ALBANY_TRILINOS_DIR:FILEPATH=install_dir \
 -D CMAKE_VERBOSE_MAKEFILE:BOOL=OFF \
 -D ENABLE_LCM:BOOL=ON \
 -D ENABLE_QCAD:BOOL=OFF \
 -D ENABLE_MOR:BOOL=OFF \
 -D ENABLE_SG_MP:BOOL=OFF \
 -D ENABLE_FELIX:BOOL=OFF \
 -D ENABLE_LAME:BOOL=OFF \
 -D ENABLE_LAMENT:BOOL=OFF \
 -D ENABLE_CHECK_FPE:BOOL=OFF \
  package_dir
