#!/bin/bash

sed -e '/CDASH-TRILINOS-FILE.TXT/r /lustre/orion/cli193/scratch/jwatkins/nightlyCDashFrontier/cdash-frag.txt' -e '/CDASH-TRILINOS-FILE.TXT/d' /lustre/orion/cli193/scratch/jwatkins/nightlyCDashFrontier/ctest_nightly_trilinos_tmp.cmake >& /lustre/orion/cli193/scratch/jwatkins/nightlyCDashFrontier/ctest_nightly_trilinos.cmake
