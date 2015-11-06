#!/bin/bash

#source $1 

TTT=`grep "(Failed)" /home/ikalash/nightlyCDash/nightly_log_functor.txt -c`
TTTT=`grep "(Not Run)" /home/ikalash/nightlyCDash/nightly_log_functor.txt -c`

#/bin/mail -s "Albany ($ALBANY_BRANCH): $TTT" "albany-regression@software.sandia.gov" < $ALBOUTDIR/albany_runtests.out
/bin/mail -s "Albany (master, KOKKOS_UNDER_DEVELOPMENT): $TTT tests failed, $TTTT tests not run" "ikalash@sandia.gov" < /home/ikalash/nightlyCDash/results_functor
/bin/mail -s "Albany (master, KOKKOS_UNDER_DEVELOPMENT): $TTT tests failed, $TTTT tests not run" "agsalin@sandia.gov" < /home/ikalash/nightlyCDash/results_functor
/bin/mail -s "Albany (master, KOKKOS_UNDER_DEVELOPMENT): $TTT tests failed, $TTTT tests not run" "gahanse@sandia.gov" < /home/ikalash/nightlyCDash/results_functor
/bin/mail -s "Albany (master, KOKKOS_UNDER_DEVELOPMENT): $TTT tests failed, $TTTT tests not run" "ambradl@sandia.gov" < /home/ikalash/nightlyCDash/results_functor
/bin/mail -s "Albany (master, KOKKOS_UNDER_DEVELOPMENT): $TTT tests failed, $TTTT tests not run" "ipdemes@sandia.gov" < /home/ikalash/nightlyCDash/results_functor
/bin/mail -s "Albany (master, KOKKOS_UNDER_DEVELOPMENT): $TTT tests failed, $TTTT tests not run" "jafike@sandia.gov" < /home/ikalash/nightlyCDash/results_functor
