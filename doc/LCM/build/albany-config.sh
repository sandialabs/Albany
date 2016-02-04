#!/bin/bash

if [ -f ./CMakeCache.txt ]; then
    rm CMakeCache.txt
fi

# The Trilinos Dir is the same as the PREFIX entry from the
# Trilinos configuration script

cmake \
 -D ALBANY_TRILINOS_DIR:FILEPATH=lcm_install_dir \
 -D CMAKE_CXX_FLAGS:STRING="lcm_cxx_flags" \
 -D CMAKE_VERBOSE_MAKEFILE:BOOL=OFF \
 -D ENABLE_LCM:BOOL=ON \
 -D ENABLE_ATO:BOOL=OFF \
 -D ENABLE_QCAD:BOOL=OFF \
 -D ENABLE_MOR:BOOL=OFF \
 -D ENABLE_SG:BOOL=OFF \
 -D ENABLE_ENSEMBLE:BOOL=OFF \
 -D ENABLE_FELIX:BOOL=OFF \
 -D ENABLE_LAME:BOOL=OFF \
 -D ENABLE_LAMENT:BOOL=OFF \
 -D ENABLE_CHECK_FPE:BOOL=lcm_fpe_switch \
 -D ENABLE_KOKKOS_UNDER_DEVELOPMENT:BOOL=lcm_enable_kokkos_devel \
 -D ALBANY_ENABLE_FORTRAN:BOOL=OFF \
 -D ENABLE_SLFAD:BOOL=lcm_enable_slfad \
 lcm_slfad_size \
 lcm_package_dir
