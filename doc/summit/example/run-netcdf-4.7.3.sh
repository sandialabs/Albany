#!/bin/bash

ZLIB_ROOT=
HDF5_ROOT=
PNETCDF_ROOT=
NETCDF_ROOT=

./configure CC=mpicc FC=mpifort CXX=mpicxx \
  CXXFLAGS="-I${ZLIB_ROOT}/include -I${HDF5_ROOT}/include -I${PNETCDF_ROOT}/include -O3" \
  CFLAGS="-I${ZLIB_ROOT}/include -I${HDF5_ROOT}/include -I${PNETCDF_ROOT}/include -O3" \
  LDFLAGS="-L${ZLIB_ROOT}/lib -L${HDF5_ROOT}/lib -L${PNETCDF_ROOT}/lib -O3" \
  FCFLAGS="-I${ZLIB_ROOT}/include -I${HDF5_ROOT}/include -I${PNETCDF_ROOT}/include -O3" \
  --disable-doxygen --enable-netcdf4 --enable-pnetcdf --disable-dap --disable-shared \
  --prefix=${NETCDF_ROOT}
make -j 4
make install

