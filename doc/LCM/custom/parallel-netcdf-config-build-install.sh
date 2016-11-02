#!/bin/bash

PACKAGE=parallel-netcdf
VERSION=`printf -- '%s\n' * | grep -oP "$PACKAGE-\K.*(?=\.tar\.bz2)"`
NAME=$PACKAGE-$VERSION
if [ -d "$NAME" ]; then
    rm "$NAME" -rf
fi
tar jxf $NAME.tar.bz2
cd $NAME
PATH=/usr/lib64/openmpi/bin:$PATH
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
