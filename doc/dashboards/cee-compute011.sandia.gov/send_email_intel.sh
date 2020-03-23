#!/bin/bash

#source $1 

TTT=`grep "(Failed)" /scratch/albany/nightly_log_intel-albany.txt -c`
TTTT=`grep "(Not Run)" /scratch/albany/nightly_log_intel-albany.txt -c`
TTTTT=`grep "(Timeout)" /scratch/albany/nightly_log_intel-albany.txt -c`
TT=`grep "...   Passed" /scratch/albany/nightly_log_intel-albany.txt -c`

#/bin/mail -s "Albany ($ALBANY_BRANCH): $TTT" "albany-regression@software.sandia.gov" < $ALBOUTDIR/albany_runtests.out
/bin/mail -s "AlbanyIntel, cee-compute019: $TT tests passed, $TTT tests failed, $TTTT tests not run, $TTTTT timeouts" "ikalash@sandia.gov, daibane@sandia.gov, tjfulle@sandia.gov, mperego@sandia.gov, lbertag@sandia.gov" < /scratch/albany/results_intel-albany
#/bin/mail -s "AlbanyIntel, cee-compute018: $TT tests passed, $TTT tests failed, $TTTT tests not run, $TTTTT timeouts" "ikalash@sandia.gov" < /scratch/albany/results_intel-albany
