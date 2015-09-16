#!/bin/bash

#source $1 

TTT=`grep "tests failed" /home/ikalash/Desktop/nightlyCDash/nightly_log_functor.txt`

#/bin/mail -s "Albany ($ALBANY_BRANCH): $TTT" "albany-regression@software.sandia.gov" < $ALBOUTDIR/albany_runtests.out
/bin/mail -s "Albany (master, KOKKOS_UNDER_DEVELOPMENT): $TTT" "ikalash@sandia.gov" < /home/ikalash/Desktop/nightlyCDash/results_functor
/bin/mail -s "Albany (master, KOKKOS_UNDER_DEVELOPMENT): $TTT" "agsalin@sandia.gov" < /home/ikalash/Desktop/nightlyCDash/results_functor
/bin/mail -s "Albany (master, KOKKOS_UNDER_DEVELOPMENT): $TTT" "gahanse@sandia.gov" < /home/ikalash/Desktop/nightlyCDash/results_functor
/bin/mail -s "Albany (master, KOKKOS_UNDER_DEVELOPMENT): $TTT" "ambradl@sandia.gov" < /home/ikalash/Desktop/nightlyCDash/results_functor
/bin/mail -s "Albany (master, KOKKOS_UNDER_DEVELOPMENT): $TTT" "ipdemes@sandia.gov" < /home/ikalash/Desktop/nightlyCDash/results_functor
