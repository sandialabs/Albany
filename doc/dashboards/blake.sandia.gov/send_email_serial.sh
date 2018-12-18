#!/bin/bash
  
#source $1 

TTT=`grep "Failed" results_serial -c`
TTTT=`grep "Not Run" results_serial -c`
TTTTT=`grep "Timeout" results_serial -c`
TT=`grep "...   Passed" results_serial -c`


echo "Subject: Albany, blake (KokkosNode=Serial): $TT tests passed, $TTT tests failed, $TTTT tests not run, $TTTTT timeouts" >& a
echo "" >& b
cat a b >& c
cat c results_serial >& d
mv d results_serial
rm a b c
cat results_serial | /usr/lib/sendmail -F ikalash@blake.sandia.gov -t "ikalash@sandia.gov, daibane@sandia.gov, tjfulle@sandia.gov, amota@sandia.gov, jwatkin@sandia.gov"

