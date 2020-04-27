#!/bin/bash

#source $1 

TTT=`grep "(Failed)" /home/ikalash/nightlyAlbanyCDash/albany-serial-intel-release.log -c`
TTTT=`grep "(Not Run)" /home/ikalash/nightlyAlbanyCDash/albany-serial-intel-release.log -c`
TTTTT=`grep "(Timeout)" /home/ikalash/nightlyAlbanyCDash/albany-serial-intel-release.log -c`
TT=`grep "...   Passed" /home/ikalash/nightlyAlbanyCDash/albany-serial-intel-release.log -c`

#/bin/mail -s "Albany ($ALBANY_BRANCH): $TTT" "albany-regression@software.sandia.gov" < $ALBOUTDIR/albany_runtests.out
/bin/mail -s "albany_cluster-toss3_skybridge-login5_serial-intel-release: $TT tests passed, $TTT tests failed, $TTTT tests not run, $TTTTT timeouts" "ikalash@sandia.gov, daibane@sandia.gov, tjfulle@sandia.gov, mperego@sandia.gov, lbertag@sandia.gov" < /home/ikalash/nightlyAlbanyCDash/results
