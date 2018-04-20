#!/bin/bash
  
#source $1 

TTT=`grep "Failed" /home/projects/albany/nightlyCDash/results_arm -c`
TTTT=`grep "Not Run" /home/projects/albany/nightlyCDash/results_arm -c`
TTTTT=`grep "Timeout" /home/projects/albany/nightlyCDash/results_arm -c`
TT=`grep "...   Passed" /home/projects/albany/nightlyCDash/results_arm -c`


echo "Subject: Albany (master, OpenMP KokkosNode, ARM): $TT tests passed, $TTT tests failed, $TTTT tests not run, $TTTTT timeouts" >& a
echo "" >& b
cat a b >& c
cat c results_arm >& d
mv d results_arm
rm a b c
cat results_arm | /usr/lib/sendmail -F ikalash@mayer.sandia.gov -t "ikalash@sandia.gov"

