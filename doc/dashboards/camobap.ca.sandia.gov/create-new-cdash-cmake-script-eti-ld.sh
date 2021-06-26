#!/bin/bash

sed -e '/CDASH-TRILINOS-FILE.TXT/r /nightlyCDash/cdash-frag-ld-eti.txt' -e '/CDASH-TRILINOS-FILE.TXT/d' /nightlyCDash/ctest_nightly_trilinos_eti_ld_tmp.cmake >& /nightlyCDash/ctest_nightly_trilinos_eti_ld.cmake
