#!/bin/bash

sed -e '/CDASH-TRILINOS-FILE.TXT/r /home/projects/albany/nightlyCDashWeaver/cdash-frag.txt' -e '/CDASH-TRILINOS-FILE.TXT/d' /home/projects/albany/nightlyCDashWeaver/ctest_nightly_trilinos_tmp.cmake >& /home/projects/albany/nightlyCDashWeaver/ctest_nightly_trilinos.cmake
