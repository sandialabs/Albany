#!/bin/bash

sed -e '/CDASH-TRILINOS-FILE.TXT/r /project/projectdirs/piscees/nightlyCoriCDash/cdash-frag.txt' -e '/CDASH-TRILINOS-FILE.TXT/d' /project/projectdirs/piscees/nightlyCoriCDash/ctest_nightly_trilinos_tmp.cmake >& /project/projectdirs/piscees/nightlyCoriCDash/ctest_nightly_trilinos.cmake
