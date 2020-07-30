#!/bin/bash

sed -e '/CDASH-TRILINOS-INTEL-FILE.TXT/r /projects/albany/nightlyAlbanyCDash/cdash-intel-frag.txt' -e '/CDASH-TRILINOS-INTEL-FILE.TXT/d' /projects/albany/nightlyAlbanyCDash/ctest_nightly_tmp.cmake >& /projects/albany/nightlyAlbanyCDash/ctest_nightly.cmake
