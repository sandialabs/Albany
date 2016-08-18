#!/bin/bash

PACKAGE=netcdf
VERSION=`printf -- '%s\n' * | grep -oP "$PACKAGE-\K.*(?=\.tar\.gz)"`
NAME=$PACKAGE-$VERSION
tar zxf $NAME.tar.gz
cd $NAME
PATH=/usr/lib64/openmpi/bin:$PATH
./configure --prefix=/usr/local/$NAME \
            CC=mpicc \
            CPPFLAGS="-I/usr/include/openmpi-x86_64/include -I/usr/local/parallel-netcdf/include" \
            LDFLAGS="-L/usr/lib64/openmpi/lib -L/usr/local/parallel-netcdf/lib" \
            --enable-shared --disable-dap --enable-pnetcdf
make -j 4
