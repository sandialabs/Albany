#!/bin/bash

#source $1 

TTT=`grep "(Failed)" /home/ikalash/nightlyAlbanyCDash/nightly_log_cismAlbany.txt -c`
TTTT=`grep "(Not Run)" /home/ikalash/nightlyAlbanyCDash/nightly_log_cismAlbany.txt -c`
TTTTT=`grep "(Timeout)" /home/ikalash/nightlyAlbanyCDash/nightly_log_cismAlbany.txt -c`
TT=`grep "...   Passed" /home/ikalash/nightlyAlbanyCDash/nightly_log_cismAlbany.txt -c`

/bin/mail -s "IKTCismAlbany, mockba.ca.sandia.gov: $TT tests passed, $TTT tests failed, $TTTT tests not run, $TTTTT timeouts" "ikalash@sandia.gov, lbertag@sandia.gov, mperego@sandia.gov" < /home/ikalash/nightlyAlbanyCDash/results_cismAlbany
