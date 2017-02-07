#!/bin/bash

#source $1 

TTT=`grep "(Failed)" ${WORKSPACE}/nightly_log_rideAlbany.txt -c`
TTTT=`grep "(Not Run)" ${WORKSPACE}/nightly_log_rideAlbany.txt -c`
TTTTT=`grep "(Timeout)" ${WORKSPACE}/nightly_log_rideAlbany.txt -c`
TT=`grep "...   Passed" ${WORKSPACE}/nightly_log_rideAlbany.txt -c`

/bin/mail -s "Albany (master, CUDA, KOKKOS_UNDER_DEVELOPMENT): $TT tests passed, $TTT tests failed, $TTTT tests not run, $TTTTT timeouts" "ikalash@sandia.gov, mperego@sandia.gov, agsalin@sandia.gov, jwatkin@sandia.gov, amota@sandia.gov, gahanse@sandia.gov, daibane@sandia.gov" < ${WORKSPACE}/results_cuda

