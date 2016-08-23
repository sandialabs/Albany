#!/bin/bash

PACKAGE=parallel-netcdf
VERSION=`printf -- '%s\n' * | grep -oP "$PACKAGE-\K.*(?=\.tar\.bz2)"`
NAME=$PACKAGE-$VERSION
tar jxf $NAME.tar.bz2
cd $NAME
PATH=/usr/lib64/openmpi/bin:$PATH
./configure --prefix=/usr/local/$NAME \
            --disable-fortran \
            CFLAGS="-fPIC -O3" \
            CXXFLAGS="-fPIC -O3" \
            FFLAGS="-fPIC -O3" \
            F90LAGS="-fPIC -O3"
make -j 4
