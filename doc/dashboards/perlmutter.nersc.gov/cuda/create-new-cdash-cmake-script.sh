#!/bin/bash

sed -e '/CDASH-TRILINOS-FILE.TXT/r /pscratch/sd/m/mcarlson/biweeklyCDashPerlmutter-cuda/cdash-frag.txt' -e '/CDASH-TRILINOS-FILE.TXT/d' /pscratch/sd/m/mcarlson/biweeklyCDashPerlmutter-cuda/ctest_nightly_trilinos_tmp.cmake >& /pscratch/sd/m/mcarlson/biweeklyCDashPerlmutter-cuda/ctest_nightly_trilinos.cmake
