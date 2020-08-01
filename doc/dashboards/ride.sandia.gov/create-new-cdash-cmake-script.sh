#!/bin/bash

sed -e '/CDASH-TRILINOS-FILE.TXT/r cdash-frag.txt' -e '/CDASH-TRILINOS-FILE.TXT/d' ctest_nightly_trilinos_tmp.cmake >& ctest_nightly_trilinos.cmake
