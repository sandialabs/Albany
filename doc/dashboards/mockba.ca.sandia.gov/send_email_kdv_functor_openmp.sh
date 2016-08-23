#!/bin/bash

#source $1 

TTT=`grep "(Failed)" /home/ikalash/Trilinos_Albany/nightlyCDash/nightly_log_kdv_functor_openmp.txt -c`
TTTT=`grep "(Not Run)" /home/ikalash/Trilinos_Albany/nightlyCDash/nightly_log_kdv_functor_openmp.txt -c`
TTTTT=`grep "Timeouts" /home/ikalash/Trilinos_Albany/nightlyCDash/nightly_log_kdv_functor_openmp.txt -c`

#/bin/mail -s "Albany ($ALBANY_BRANCH): $TTT" "albany-regression@software.sandia.gov" < $ALBOUTDIR/albany_runtests.out
/bin/mail -s "Albany (kdv branch, functor OpenMP node): $TTT tests failed, $TTTT tests not run, $TTTTT timeouts" "ikalash@sandia.gov" < /home/ikalash/Trilinos_Albany/nightlyCDash/results_kdv_functor_openmp
#/bin/mail -s "Albany (kdv branch, functor OpenMP node): $TTT tests failed, $TTTT tests not run, $TTTTT timeouts" "agsalin@sandia.gov" < /home/ikalash/Trilinos_Albany/nightlyCDash/results_kdv_functor_openmp
#/bin/mail -s "Albany (kdv branch, functor OpenMP node): $TTT tests failed, $TTTT tests not run, $TTTTT timeouts" "jwatkin@sandia.gov" < /home/ikalash/Trilinos_Albany/nightlyCDash/results_kdv_functor_openmp
#/bin/mail -s "Albany (kdv branch, functor OpenMP node): $TTT tests failed, $TTTT tests not run, $TTTTT timeouts" "mperego@sandia.gov" < /home/ikalash/Trilinos_Albany/nightlyCDash/results_kdv_functor_openmp
