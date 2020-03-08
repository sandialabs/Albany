#!/bin/bash
  
#source $1 

TTT=`grep "(Failed)" nightly_log_blakeAlbanyOpenMP.txt -c`
TTTT=`grep "Not Run" nightly_log_blakeAlbanyOpenMP.txt -c`
TTTTT=`grep "\*Timeout" nightly_log_blakeAlbanyOpenMP.txt -c`
TT=`grep "...   Passed" nightly_log_blakeAlbanyOpenMP.txt -c`


echo "Subject: Albany, blake (KokkosNode=OpenMP): $TT tests passed, $TTT tests failed, $TTTT tests not run, $TTTTT timeouts" >& a
echo "" >& b
cat a b >& c
cat c results_openmp >& d
mv d results_openmp
rm a b c
cat nightly_log_blakeAlbanyOpenMP.txt | /usr/lib/sendmail -F ikalash@blake.sandia.gov -t "ikalash@sandia.gov, daibane@sandia.gov, tjfulle@sandia.gov, jwatkin@sandia.gov"
#cat results_openmp | /usr/lib/sendmail -F ikalash@blake.sandia.gov -t "ikalash@sandia.gov"

