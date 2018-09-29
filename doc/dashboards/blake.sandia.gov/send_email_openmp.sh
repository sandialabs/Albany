#!/bin/bash
  
#source $1 

TTT=`grep "Failed" results_openmp -c`
TTTT=`grep "Not Run" results_openmp -c`
TTTTT=`grep "Timeout" results_openmp -c`
TT=`grep "...   Passed" results_openmp -c`


echo "Subject: Albany (master, OpenMP, Skylake): $TT tests passed, $TTT tests failed, $TTTT tests not run, $TTTTT timeouts" >& a
echo "" >& b
cat a b >& c
cat c results_openmp >& d
mv d results_openmp
rm a b c
cat results_openmp | /usr/lib/sendmail -F ikalash@blake.sandia.gov -t "ikalash@sandia.gov, gahanse@sandia.gov, daibane@sandia.gov, tjfulle@sandia.gov, amota@sandia.gov, jwatkin@sandia.gov"

