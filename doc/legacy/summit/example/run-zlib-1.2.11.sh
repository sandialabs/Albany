#!/bin/bash

ZLIB_ROOT=

CC=mpicc CFLAGS=-O3 ./configure --prefix=${ZLIB_ROOT}
make -j 4
make install

