#!/bin/bash

#source $1 

TTT=`grep "(Failed)" /scratch/albany/spack_ctest.out -c`
TTTT=`grep "(Not Run)" /scratch/albany/spack_ctest.out -c`
TTTTT=`grep "(Timeout)" /scratch/albany/spack_ctest.out -c`
TT=`grep "...   Passed" /scratch/albany/spack_ctest.out -c`

#/bin/mail -s "Albany ($ALBANY_BRANCH): $TTT" "albany-regression@software.sandia.gov" < $ALBOUTDIR/albany_runtests.out
#/bin/mail -s "Albany Spack Build, camobap.ca.sandia.gov: $TT tests passed, $TTT tests failed, $TTTT tests not run, $TTTTT timeouts" "ikalash@sandia.gov, daibane@sandia.gov, tjfulle@sandia.gov, mperego@sandia.gov, lbertag@sandia.gov" < /home/ikalash/nightlyAlbanyTests/results_spack
/bin/mail -s "Albany Spack Build, ascic102.sandia.gov, Clang: $TT tests passed, $TTT tests failed, $TTTT tests not run, $TTTTT timeouts" "ikalash@sandia.gov" < /scratch/albany/results_spack
