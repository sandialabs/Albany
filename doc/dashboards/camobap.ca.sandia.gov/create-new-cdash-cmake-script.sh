#!/bin/bash

sed -e '/CDASH-TRILINOS-FILE.TXT/r /nightlyCDash/cdash-frag.txt' -e '/CDASH-TRILINOS-FILE.TXT/d' /nightlyCDash/ctest_nightly_trilinos_tmp.cmake >& /nightlyCDash/ctest_nightly_trilinos.cmake
