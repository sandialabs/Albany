#!/bin/bash

# This file sets the environment in preparation for building FMDB

if [ ! -d "fmdbParallel" ]; then
  mkdir fmdbParallel
fi

export INSTALL_TARGET=$PWD/fmdbParallel
export CREATE_TARBALLS=0
export SCOREC_SVN=1

export MPIHOME=$MPI_BASE_DIR

export CXX=$MPIHOME/bin/mpicxx

export CC=$MPIHOME/bin/mpicc

export PARMETIS_HOME=/repository/usr/local/parallel/ParMetis-3.1.1
export ZOLTAN_HOME=$TRILINOS_INSTALL_DIR

# is this needed???

export PKG_CONFIG_PATH=/home/gahanse/Codes/SCOREC/fmdbParallel/lib/pkgconfig

