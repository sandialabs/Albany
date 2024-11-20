#!/bin/bash

sed -e "/CDASH-ALBANY-FILE.TXT/r /lustre/orion/cli193/scratch/mcarlson/testingCDashFrontier-rocm/cdash-albany-frag.txt" -e "/CDASH-ALBANY-FILE.TXT/d" "/lustre/orion/cli193/scratch/mcarlson/testingCDashFrontier-rocm/ctest_nightly_albany_tmp.cmake" >& /lustre/orion/cli193/scratch/mcarlson/testingCDashFrontier-rocm/ctest_nightly_albany.cmake
