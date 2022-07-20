#!/bin/bash

#source $1 

TTT=`grep "(Failed)" /home/ikalash/Trilinos_Albany/nightlyAlbanyTests/spack_ctest.out -c`
TTTT=`grep "(Not Run)" /home/ikalash/Trilinos_Albany/nightlyAlbanyTests/spack_ctest.out -c`
TTTTT=`grep "(Timeout)" /home/ikalash/Trilinos_Albany/nightlyAlbanyTests/spack_ctest.out -c`
TT=`grep "...   Passed" /home/ikalash/Trilinos_Albany/nightlyAlbanyTests/spack_ctest.out -c`

/bin/mail -s "Albany Spack Build (linux-rhel7-westmere, gcc-6.1.0): $TT tests passed, $TTT tests failed, $TTTT tests not run, $TTTTT timeouts" "ikalash@sandia.gov, mperego@sandia.gov, lbertag@sandia.gov, gahanse@sandia.gov" < /home/ikalash/Trilinos_Albany/nightlyAlbanyTests/results_spack
