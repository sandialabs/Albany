#!/bin/bash
  
#source $1 

TTT=`grep "Failed" results_arm -c`
TTTT=`grep "Not Run" results_arm -c`
TTTTT=`grep "Timeout" results_arm -c`
TT=`grep "...   Passed" results_arm -c`


echo "Subject: Albany, mayer (KokkosNode=Serial): $TT tests passed, $TTT tests failed, $TTTT tests not run, $TTTTT timeouts" >& a
echo "" >& b
cat a b >& c
cat c results_arm >& d
mv d results_arm
rm a b c
cat results_arm | /usr/lib/sendmail -F ikalash@mayer.sandia.gov -t "ikalash@sandia.gov, daibane@sandia.gov, tjfulle@sandia.gov, amota@sandia.gov, jwatkin@sandia.gov"

