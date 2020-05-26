#!/bin/bash

PNETCDF_ROOT=

./configure MPICC=mpicc MPICXX=mpicxx MPIF77=mpif77 MPIF90=mpif90 \
  CXXFLAGS="-O3" CFLAGS="-O3" F77FLAGS="-O3" F90FLAGS="-O3" \
  --enable-fortran --build=ppc64 \
  --prefix=${PNETCDF_ROOT}
make -j 4
make install

