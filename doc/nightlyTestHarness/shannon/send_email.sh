#!/bin/bash


SCRIPTDIR=/home/ikalash/nightlyAlbanyTests/
TRILINOS_BRANCH=develop
ALBANY_BRANCH=DynRankViewIntrepid2Refactor

mail -s "Shannon Trilinos ($TRILINOS_BRANCH) and Albany ($ALBANY_BRANCH) CUDA test results" "ikalash@sandia.gov, jwatkin@sandia.gov, asalin@sandia.gov, mperego@sandia.gov" < $SCRIPTDIR/tests.out

