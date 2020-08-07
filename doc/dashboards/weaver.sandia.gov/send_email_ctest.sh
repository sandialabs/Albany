#!/bin/bash

#source $1 

TTT=`grep "(Failed)" nightly_log_weaverAlbany.txt -c`
TTTT=`grep "(Not Run)" nightly_log_weaverAlbany.txt -c`
TTTTT=`grep "(Timeout)" nightly_log_weaverAlbany.txt -c`
TT=`grep "...   Passed" nightly_log_weaverAlbany.txt -c`

/bin/mail -s "Albany, weaver (KokkosNode=CUDA): $TT tests passed, $TTT tests failed, $TTTT tests not run, $TTTTT timeouts" "ikalash@sandia.gov, mperego@sandia.gov, jwatkin@sandia.gov, daibane@sandia.gov, tjfulle@sandia.gov" < results_cuda

