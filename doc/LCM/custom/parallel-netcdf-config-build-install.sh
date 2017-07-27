#!/bin/bash

PACKAGE=parallel-netcdf
VERSION=`printf -- '%s\n' * | grep -oP "$PACKAGE-\K.*(?=\.tar\.gz)"`
NAME=$PACKAGE-$VERSION
if [ -d "$NAME" ]; then
    rm "$NAME" -rf
fi
tar zxf $NAME.tar.gz
cd $NAME
PATH=/usr/lib64/openmpi/bin:$PATH
CC=`which mpicc`
MPICC=`which mpicc`
INSTALL_DIR=/usr/local/$NAME
./configure --prefix=$INSTALL_DIR \
            --disable-fortran \
            CFLAGS="-fPIC -O3" \
            CXXFLAGS="-fPIC -O3" \
            FFLAGS="-fPIC -O3" \
            F90LAGS="-fPIC -O3"
make -j 4
if [ -d "$INSTALL_DIR" ]; then
    sudo rm "$INSTALL_DIR" -rf
fi
sudo make install
