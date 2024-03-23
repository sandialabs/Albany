#!/bin/bash

#source $1 

TTT=`grep "(Failed)" /projects/albany/nightlySpackBuild/spack_ctest.out -c`
TTTT=`grep "(Not Run)" /projects/albany/nightlySpackBuild/spack_ctest.out -c`
TTTTT=`grep "(Timeout)" /projects/albany/nightlySpackBuild/spack_ctest.out -c`
TT=`grep "...   Passed" /projects/albany/nightlySpackBuild/spack_ctest.out -c`


echo "Subject: Albany Spack Build, cee-compute006 (linux-rhel7, gcc-10.1.0): $TT tests passed, $TTT tests failed, $TTTT tests not run, $TTTTT timeouts" >& a
echo "" >& b
cat a b >& c
cat c results_spack >& d
rm a b c
mv d results_spack
cat results_spack | /usr/sbin/sendmail -F ikalash@camobap.ca.sandia.gov -t "ikalash@sandia.gov, mperego@sandia.gov, lbertag@sandia.gov"
