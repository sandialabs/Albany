#!/bin/bash

#source $1 

TTT=`grep "(Failed)" /nightlyAlbanyTests/spack_ctest.out -c`
TTTT=`grep "(Not Run)" /nightlyAlbanyTests/spack_ctest.out -c`
TTTTT=`grep "(Timeout)" /nightlyAlbanyTests/spack_ctest.out -c`
TT=`grep "...   Passed" /nightlyAlbanyTests/spack_ctest.out -c`

#/bin/mail -s "Albany ($ALBANY_BRANCH): $TTT" "albany-regression@software.sandia.gov" < $ALBOUTDIR/albany_runtests.out
#/bin/mail -s "Albany Spack Build, camobap.ca.sandia.gov: $TT tests passed, $TTT tests failed, $TTTT tests not run, $TTTTT timeouts" "ikalash@sandia.gov, daibane@sandia.gov, tjfulle@sandia.gov, mperego@sandia.gov, lbertag@sandia.gov" < /nightlyAlbanyTests/results_spack
/bin/mail -s "Albany Spack Build (linux-fedora32-haswell, gcc-10.1.1): $TT tests passed, $TTT tests failed, $TTTT tests not run, $TTTTT timeouts" "ikalash@sandia.gov, mperego@sandia.gov, lbertag@sandia.gov, gahanse@sandia.gov" < /nightlyAlbanyTests/results_spack
#/bin/mail -s "Albany Spack Build (linux-fedora32-haswell, gcc-10.1.1): $TT tests passed, $TTT tests failed, $TTTT tests not run, $TTTTT timeouts" "ikalash@sandia.gov" < /nightlyAlbanyTests/results_spack
