#!/bin/bash

BOOST_DIR=

./bootstrap.sh --prefix=${BOOST_DIR} --with-toolset=gcc

./b2 -j 16 --without-python --without-stacktrace install

