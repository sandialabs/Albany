#!/bin/bash
  
#source $1 

TTT=`grep "(Failed)" nightly_log_mayerAlbany.txt -c`
TTTT=`grep "Not Run" nightly_log_mayerAlbany.txt -c`
TTTTT=`grep "\*Timeout" nightly_log_mayerAlbany.txt -c`
TT=`grep "...   Passed" nightly_log_mayerAlbany.txt -c`


echo "Subject: Albany, mayer (KokkosNode=Serial): $TT tests passed, $TTT tests failed, $TTTT tests not run, $TTTTT timeouts" >& a
echo "" >& b
cat a b >& c
cat c results_arm >& d
mv d results_arm
rm a b c
cat results_arm | /usr/lib/sendmail -F ikalash@mayer.sandia.gov -t "ikalash@sandia.gov, daibane@sandia.gov, tjfulle@sandia.gov, jwatkin@sandia.gov"
#cat results_arm | /usr/lib/sendmail -F ikalash@mayer.sandia.gov -t "ikalash@sandia.gov"

