#!/bin/bash
  
#source $1 

TTT=`grep "Failed" /home/projects/albany/nightlyCDashAlbanyJenkins/results_arm -c`
TTTT=`grep "Not Run" /home/projects/albany/nightlyCDashAlbanyJenkins/results_arm -c`
TTTTT=`grep "Timeout" /home/projects/albany/nightlyCDashAlbanyJenkins/results_arm -c`
TT=`grep "...   Passed" /home/projects/albany/nightlyCDashAlbanyJenkins/results_arm -c`


echo "Subject: Albany (master, Serial KokkosNode, ARM): $TT tests passed, $TTT tests failed, $TTTT tests not run, $TTTTT timeouts" >& a
echo "" >& b
cat a b >& c
cat c results_arm >& d
mv d results_arm
rm a b c
cat results_arm | /usr/lib/sendmail -F ikalash@mayer.sandia.gov -t "ikalash@sandia.gov, gahanse@sandia.gov, daibane@sandia.gov, tjfulle@sandia.gov, amota@sandia.gov, jwatkin@sandia.gov"

