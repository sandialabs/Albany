#!/bin/bash

#source $1 

TTT=`grep "(Failed)" /nightlySpackBuild/spack_ctest.out -c`
TTTT=`grep "(Not Run)" /nightlySpackBuild/spack_ctest.out -c`
TTTTT=`grep "(Timeout)" /nightlySpackBuild/spack_ctest.out -c`
TT=`grep "...   Passed" /nightlySpackBuild/spack_ctest.out -c`


echo "Subject: Albany Spack Build, camobap (linux-rhel8, gcc-11.1.0): $TT tests passed, $TTT tests failed, $TTTT tests not run, $TTTTT timeouts" >& a
echo "" >& b
cat a b >& c
cat c results_spack >& d
rm a b c
mv d results_spack
cat results_spack | /usr/sbin/sendmail -F ikalash@camobap.ca.sandia.gov -t "ikalash@sandia.gov, mperego@sandia.gov, lbertag@sandia.gov"
