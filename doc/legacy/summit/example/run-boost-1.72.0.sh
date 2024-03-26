#!/bin/bash

BOOST_ROOT=

./bootstrap.sh --prefix=${BOOST_ROOT} --with-toolset=gcc

./b2 -j 4 --without-python --without-stacktrace link=static install

