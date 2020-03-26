#!/bin/bash

#source $1 

TTT=`grep "(Failed)" /home/ikalash/nightlyAlbanyCDash/nightly_log.txt -c`
TTTT=`grep "(Not Run)" /home/ikalash/nightlyAlbanyCDash/nightly_log.txt -c`
TTTTT=`grep "(Timeout)" /home/ikalash/nightlyAlbanyCDash/nightly_log.txt -c`
TT=`grep "...   Passed" /home/ikalash/nightlyAlbanyCDash/nightly_log.txt -c`

#/bin/mail -s "Albany ($ALBANY_BRANCH): $TTT" "albany-regression@software.sandia.gov" < $ALBOUTDIR/albany_runtests.out
/bin/mail -s "IKTAlbanyMockba, mockba.ca.sandia.gov: $TT tests passed, $TTT tests failed, $TTTT tests not run, $TTTTT timeouts" "ikalash@sandia.gov, daibane@sandia.gov, tjfulle@sandia.gov, mperego@sandia.gov, lbertag@sandia.gov" < /home/ikalash/nightlyAlbanyCDash/results
