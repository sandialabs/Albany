#!/bin/bash

#source $1 

TTT=`grep "(Failed)" nightly_log_coriCismAlbanyRun.txt -c`
TTTT=`grep "(Not Run)" nightly_log_coriCismAlbanyRun.txt -c`
TTTTT=`grep "(Timeout)" nightly_log_coriCismAlbanyRun.txt -c`
TT=`grep "...   Passed" nightly_log_coriCismAlbanyRun.txt -c`

#/bin/mail -s "Albany ($ALBANY_BRANCH): $TTT" "albany-regression@software.sandia.gov" < $ALBOUTDIR/albany_runtests.out
/bin/mail -s "CoriCismAlbany, cori.nersc.gov: $TT tests passed, $TTT tests failed, $TTTT tests not run, $TTTTT timeouts" "ikalash@sandia.gov, lbertag@sandia.gov, mperego@sandia.gov" -F "Irina Tezaur" < results_coriCismAlbany
