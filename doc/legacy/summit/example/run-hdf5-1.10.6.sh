#!/bin/bash

ZLIB_ROOT=
HDF5_ROOT=

./configure CC=mpicc FC=mpifort CXX=mpicxx CXXFLAGS="-O3" CFLAGS="-O3" FCFLAGS="-O3" \
  --enable-fortran --disable-shared --enable-parallel --enable-static --build=ppc64 \
  --with-zlib=${ZLIB_ROOT} --prefix=${HDF5_ROOT}
make -j 4
make install

