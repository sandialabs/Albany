#!/bin/bash
  
#source $1 

TTT=`grep "(Failed)" nightly_log_mayerAlbany.txt -c`
TTTT=`grep "Not Run" nightly_log_mayerAlbany.txt -c`
TTTTT=`grep "\*Timeout" nightly_log_mayerAlbany.txt -c`
TT=`grep "...   Passed" nightly_log_mayerAlbany.txt -c`


echo "Subject: Albany, mayer (KokkosNode=Serial): $TT tests passed, $TTT tests failed, $TTTT tests not run, $TTTTT timeouts" >& /mscratch/albany/a
echo "" >& /mscratch/albany/b
cat /mscratch/albany/a /mscratch/albany/b >& /mscratch/albany/c
cat /mscratch/albany/c /mscratch/albany/results_arm >& /mscratch/albany/d
mv /mscratch/albany/d /mscratch/albany/results_arm
rm /mscratch/albany/a /mscratch/albany/b /mscratch/albany/c
cat /mscratch/albany/results_arm | /usr/lib/sendmail -F ikalash@mayer.sandia.gov -t "ikalash@sandia.gov, daibane@sandia.gov, tjfulle@sandia.gov, amota@sandia.gov, jwatkin@sandia.gov"
#cat results_arm | /usr/lib/sendmail -F ikalash@mayer.sandia.gov -t "ikalash@sandia.gov"

