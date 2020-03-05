#!/bin/bash

#source $1 

TTT=`grep "(Failed)" /home/ikalash/nightlyAlbanyTests/spack_ctest.out -c`
TTTT=`grep "(Not Run)" /home/ikalash/nightlyAlbanyTests/spack_ctest.out -c`
TTTTT=`grep "(Timeout)" /home/ikalash/nightlyAlbanyTests/spack_ctest.out -c`
TT=`grep "...   Passed" /home/ikalash/nightlyAlbanyTests/spack_ctest.out -c`

/bin/mail -s "Albany Spack Build (linux-fedora31-haswell, gcc-9.2.1): $TT tests passed, $TTT tests failed, $TTTT tests not run, $TTTTT timeouts" "ikalash@sandia.gov, mperego@sandia.gov, lbertag@sandia.gov" < /home/ikalash/nightlyAlbanyTests/results_spack
