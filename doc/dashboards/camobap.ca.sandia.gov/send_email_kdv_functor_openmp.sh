#!/bin/bash

#source $1 

TTT=`grep "(Failed)" /home/ikalash/nightlyCDash/nightly_log_kdv_functor_openmp.txt -c`
TTTT=`grep "(Not Run)" /home/ikalash/nightlyCDash/nightly_log_kdv_functor_openmp.txt -c`
TTTTT=`grep "Timeouts" /home/ikalash/nightlyCDash/nightly_log_kdv_functor_openmp.txt -c`

#/bin/mail -s "Albany ($ALBANY_BRANCH): $TTT" "albany-regression@software.sandia.gov" < $ALBOUTDIR/albany_runtests.out
/bin/mail -s "Albany (DynRankViewIntrepid2Refactor, OpenMP, KOKKOS_UNDER_DEVELOPMENT): $TTT tests failed, $TTTT tests not run, $TTTTT timeouts" "ikalash@sandia.gov, agsalin@sandia.gov, jwatkin@sandia.gov, mperego@sandia.gov" < /home/ikalash/nightlyCDash/results_kdv_functor_openmp
