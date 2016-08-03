#!/bin/bash

#source $1 

TTT=`grep "(Failed)" /home/ikalash/Trilinos_Albany/nightlyCDash/nightly_log_kdv_no_functor.txt -c`
TTTT=`grep "(Not Run)" /home/ikalash/Trilinos_Albany/nightlyCDash/nightly_log_kdv_no_functor.txt -c`
TTTTT=`grep "Timeouts" /home/ikalash/Trilinos_Albany/nightlyCDash/nightly_log_kdv_no_functor.txt -c`

#/bin/mail -s "Albany ($ALBANY_BRANCH): $TTT" "albany-regression@software.sandia.gov" < $ALBOUTDIR/albany_runtests.out
/bin/mail -s "Albany (DynRankViewIntrepid2Refactor): $TTT tests failed, $TTTT tests not run, $TTTTT timeouts" "ikalash@sandia.gov, agsalin@sandia.gov, jwatkin@sandia.gov, mperego@sandia.gov" < /home/ikalash/Trilinos_Albany/nightlyCDash/results_kdv_no_functor
