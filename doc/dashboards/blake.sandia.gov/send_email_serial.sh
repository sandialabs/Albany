#!/bin/bash
  
#source $1 

TTT=`grep "(Failed)" nightly_log_blakeAlbanySerial.txt -c`
TTTT=`grep "Not Run" nightly_log_blakeAlbanySerial.txt -c`
TTTTT=`grep "\*Timeout" nightly_log_blakeAlbanySerial.txt -c`
TT=`grep "...   Passed" nightly_log_blakeAlbanySerial.txt -c`


echo "Subject: Albany, blake (KokkosNode=Serial): $TT tests passed, $TTT tests failed, $TTTT tests not run, $TTTTT timeouts" >& a
echo "" >& b
cat a b >& c
cat c results_serial >& d
mv d results_serial
rm a b c
cat results_serial | /usr/lib/sendmail -F ikalash@blake.sandia.gov -t "ikalash@sandia.gov, daibane@sandia.gov, tjfulle@sandia.gov, jwatkin@sandia.gov"
#cat results_serial | /usr/lib/sendmail -F ikalash@blake.sandia.gov -t "ikalash@sandia.gov"

