#!/bin/bash

#source $1 

TTT=`grep "(Failed)" nightly_log_watermanAlbany.txt -c`
TTTT=`grep "(Not Run)" nightly_log_watermanAlbany.txt -c`
TTTTT=`grep "(Timeout)" nightly_log_watermanAlbany.txt -c`
TT=`grep "...   Passed" nightly_log_watermanAlbany.txt -c`

/bin/mail -s "Albany, waterman (KokkosNode=CUDA): $TT tests passed, $TTT tests failed, $TTTT tests not run, $TTTTT timeouts" "ikalash@sandia.gov, mperego@sandia.gov, jwatkin@sandia.gov, daibane@sandia.gov, tjfulle@sandia.gov" < results_cuda
#/bin/mail -s "Albany (master, Waterman CUDA+V100+P9, KOKKOS_UNDER_DEVELOPMENT): $TT tests passed, $TTT tests failed, $TTTT tests not run, $TTTTT timeouts" "ikalash@sandia.gov" < results_cuda

