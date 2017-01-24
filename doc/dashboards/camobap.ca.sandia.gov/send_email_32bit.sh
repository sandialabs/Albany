#!/bin/bash

#source $1 

TTT=`grep "(Failed)" /home/ikalash/nightlyCDash/nightly_log_32bit.txt -c`
TTTT=`grep "(Not Run)" /home/ikalash/nightlyCDash/nightly_log_32bit.txt -c`
TTTTT=`grep "(Timeout)" /home/ikalash/nightlyCDash/nightly_log_32bit.txt -c`

#/bin/mail -s "Albany ($ALBANY_BRANCH): $TTT" "albany-regression@software.sandia.gov" < $ALBOUTDIR/albany_runtests.out
/bin/mail -s "Albany (master, 32bit): $TTT tests failed, $TTTT tests not run, $TTTTT timeouts" "ikalash@sandia.gov, agsalin@sandia.gov, gahanse@sandia.gov, daibane@sandia.gov" < /home/ikalash/nightlyCDash/results_32bit
