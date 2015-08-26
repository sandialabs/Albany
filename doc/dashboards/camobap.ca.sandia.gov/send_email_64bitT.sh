#!/bin/bash

#source $1 

TTT=`grep "tests failed" /home/ikalash/Desktop/nightlyCDash/nightly_log_64bitT.txt`

#/bin/mail -s "Albany ($ALBANY_BRANCH): $TTT" "albany-regression@software.sandia.gov" < $ALBOUTDIR/albany_runtests.out
/bin/mail -s "Albany (master, 64bit, AlbanyT only): $TTT" "ikalash@sandia.gov" < /home/ikalash/Desktop/nightlyCDash/results_64bitT
/bin/mail -s "Albany (master, 64bit, AlbanyT only): $TTT" "agsalin@sandia.gov" < /home/ikalash/Desktop/nightlyCDash/results_64bitT
/bin/mail -s "Albany (master, 64bit, AlbanyT only): $TTT" "gahanse@sandia.gov" < /home/ikalash/Desktop/nightlyCDash/results_64bitT
/bin/mail -s "Albany (master, 64bit, AlbanyT only): $TTT" "ambradl@sandia.gov" < /home/ikalash/Desktop/nightlyCDash/results_64bitT
