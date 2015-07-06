#!/bin/bash

#source $1 

TTT=`grep "tests failed" /home/ikalash/Desktop/nightlyCDash/nightly_log_32bitT.txt`

#/bin/mail -s "Albany ($ALBANY_BRANCH): $TTT" "albany-regression@software.sandia.gov" < $ALBOUTDIR/albany_runtests.out
/bin/mail -s "Albany (master, 32bit, AlbanyT only): $TTT" "ikalash@sandia.gov" < /home/ikalash/Desktop/nightlyCDash/results_32bitT
/bin/mail -s "Albany (master, 32bit, AlbanyT only): $TTT" "agsalin@sandia.gov" < /home/ikalash/Desktop/nightlyCDash/results_32bitT
/bin/mail -s "Albany (master, 32bit, AlbanyT only): $TTT" "gahanse@sandia.gov" < /home/ikalash/Desktop/nightlyCDash/results_32bitT
/bin/mail -s "Albany (master, 32bit, AlbanyT only): $TTT" "ambradl@sandia.gov" < /home/ikalash/Desktop/nightlyCDash/results_32bitT
