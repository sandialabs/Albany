#!/bin/bash

PACKAGE=netcdf
VERSION=`printf -- '%s\n' * | grep -oP "$PACKAGE-\K.*(?=\.tar\.gz)"`
NAME=$PACKAGE-$VERSION
if [ -d "$NAME" ]; then
    rm "$NAME" -rf
fi
tar zxf $NAME.tar.gz
cd $NAME
cd include
sed -i -e "s/#define NC_MAX_DIMS.*$/#define NC_MAX_DIMS     65536/g;" netcdf.h
sed -i -e "s/#define NC_MAX_ATTRS.*$/#define NC_MAX_ATTRS    8192/g;" netcdf.h
sed -i -e "s/#define NC_MAX_VARS.*$/#define NC_MAX_VARS     524288/g;" netcdf.h
sed -i -e "s/#define NC_MAX_NAME.*$/#define NC_MAX_NAME     256/g;" netcdf.h
sed -i -e "s/#define NC_MAX_VAR_DIMS.*$/#define NC_MAX_VAR_DIMS 8/g;" netcdf.h
cd ..
PATH=/usr/lib64/openmpi/bin:$PATH
INSTALL_DIR=/usr/local/$NAME
./configure --prefix=$INSTALL_DIR \
            CC=mpicc \
            CPPFLAGS="-I/usr/include/openmpi-x86_64/include -I/usr/local/parallel-netcdf/include" \
            LDFLAGS="-L/usr/lib64/openmpi/lib -L/usr/local/parallel-netcdf/lib" \
            --enable-shared --disable-dap --enable-pnetcdf
make -j 4
if [ -d "$INSTALL_DIR" ]; then
    sudo rm "$INSTALL_DIR" -rf
fi
sudo make install
