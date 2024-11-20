#!/bin/bash

sed -e '/CDASH-TRILINOS-FILE.TXT/r /lustre/orion/cli193/scratch/mcarlson/testingCDashFrontier-rocm/cdash-frag.txt' -e '/CDASH-TRILINOS-FILE.TXT/d' /lustre/orion/cli193/scratch/mcarlson/testingCDashFrontier-rocm/ctest_nightly_trilinos_tmp.cmake >& /lustre/orion/cli193/scratch/mcarlson/testingCDashFrontier-rocm/ctest_nightly_trilinos.cmake
