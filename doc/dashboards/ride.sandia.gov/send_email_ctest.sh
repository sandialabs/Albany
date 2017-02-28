#!/bin/bash

#source $1 

TTT=`grep "(Failed)" nightly_log_rideAlbany.txt -c`
TTTT=`grep "(Not Run)" nightly_log_rideAlbany.txt -c`
TTTTT=`grep "(Timeout)" nightly_log_rideAlbany.txt -c`
TT=`grep "...   Passed" nightly_log_rideAlbany.txt -c`

/bin/mail -s "Albany (master, CUDA, KOKKOS_UNDER_DEVELOPMENT): $TT tests passed, $TTT tests failed, $TTTT tests not run, $TTTTT timeouts" "ikalash@sandia.gov, mperego@sandia.gov, agsalin@sandia.gov, jwatkin@sandia.gov, amota@sandia.gov, gahanse@sandia.gov, daibane@sandia.gov" < results_cuda

