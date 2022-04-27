#!/bin/bash

#source $1 

TTT=`grep "(Failed)" /projects/albany/nightlySpackBuild/spack_ctest.out -c`
TTTT=`grep "(Not Run)" /projects/albany/nightlySpackBuild/spack_ctest.out -c`
TTTTT=`grep "(Timeout)" /projects/albany/nightlySpackBuild/spack_ctest.out -c`
TT=`grep "...   Passed" /projects/albany/nightlySpackBuild/spack_ctest.out -c`

#/bin/mail -s "Albany ($ALBANY_BRANCH): $TTT" "albany-regression@software.sandia.gov" < $ALBOUTDIR/albany_runtests.out
#/bin/mail -s "Albany Spack Build, camobap.ca.sandia.gov: $TT tests passed, $TTT tests failed, $TTTT tests not run, $TTTTT timeouts" "albany@sandia.gov, daibane@sandia.gov, tjfulle@sandia.gov, mperego@sandia.gov, lbertag@sandia.gov" < /projects/albany/nightlySpackBuild/results_spack
/bin/mail -s "Albany Spack Build (linux-rhel7-haswell, gcc-9.2.0): $TT tests passed, $TTT tests failed, $TTTT tests not run, $TTTTT timeouts" "ikalash@sandia.gov, mperego@sandia.gov, lbertag@sandia.gov" < /projects/albany/nightlySpackBuild/results_spack
#/bin/mail -s "Albany Spack Build (linux-rhel7-haswell, gcc-10.1.1): $TT tests passed, $TTT tests failed, $TTTT tests not run, $TTTTT timeouts" "ikalash@sandia.gov" < /projects/albany/nightlySpackBuild/results_spack
