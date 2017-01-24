#!/bin/bash

#source $1 

TTT=`grep "(Failed)" /home/ikalash/nightlyCDash/nightly_log_functor.txt -c`
TTTT=`grep "(Not Run)" /home/ikalash/nightlyCDash/nightly_log_functor.txt -c`
TTTTT=`grep "(Timeout)" /home/ikalash/nightlyCDash/nightly_log_functor.txt -c`

#/bin/mail -s "Albany ($ALBANY_BRANCH): $TTT" "albany-regression@software.sandia.gov" < $ALBOUTDIR/albany_runtests.out
/bin/mail -s "Albany (master, KOKKOS_UNDER_DEVELOPMENT): $TTT tests failed, $TTTT tests not run, $TTTTT timeouts" "ikalash@sandia.gov, agsalin@sandia.gov, gahanse@sandia.gov, jafike@sandia.gov, jwatkin@sandia.gov, daibane@sandia.gov" < /home/ikalash/nightlyCDash/results_functor
