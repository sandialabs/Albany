#!/bin/bash

#source $1 

TTT=`grep "(Failed)" /mnt/encrypted_sdc1/nightlyCDash/nightly_log_kokkosnode_openmp.txt -c`
TTTT=`grep "(Not Run)" /mnt/encrypted_sdc1/nightlyCDash/nightly_log_kokkosnode_openmp.txt -c`
TTTTT=`grep "(Timeout)" /mnt/encrypted_sdc1/nightlyCDash/nightly_log_kokkosnode_openmp.txt -c`
TT=`grep "...   Passed" /mnt/encrypted_sdc1/nightlyCDash/nightly_log_kokkosnode_openmp.txt -c`

#/bin/mail -s "Albany ($ALBANY_BRANCH): $TTT" "albany-regression@software.sandia.gov" < $ALBOUTDIR/albany_runtests.out
/bin/mail -s "IKTAlbanyFunctorOpenMP, camobap.ca.sandia.gov: $TT tests passed, $TTT tests failed, $TTTT tests not run, $TTTTT timeouts" "ikalash@sandia.gov, jwatkin@sandia.gov, daibane@sandia.gov, tjfulle@sandia.gov, mperego@sandia.gov, lbertag@sandia.gov" < /mnt/encrypted_sdc1/nightlyCDash/results_functor_openMP
