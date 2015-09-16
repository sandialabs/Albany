#!/bin/bash

#source $1 

TTT=`grep "tests failed" /home/ikalash/Desktop/nightlyCDash/nightly_log_64bit.txt`

#/bin/mail -s "Albany ($ALBANY_BRANCH): $TTT" "albany-regression@software.sandia.gov" < $ALBOUTDIR/albany_runtests.out
/bin/mail -s "Albany (master, 64bit): $TTT" "ikalash@sandia.gov" < /home/ikalash/Desktop/nightlyCDash/results_64bit
/bin/mail -s "Albany (master, 64bit): $TTT" "agsalin@sandia.gov" < /home/ikalash/Desktop/nightlyCDash/results_64bit
/bin/mail -s "Albany (master, 64bit): $TTT" "gahanse@sandia.gov" < /home/ikalash/Desktop/nightlyCDash/results_64bit
/bin/mail -s "Albany (master, 64bit): $TTT" "ambradl@sandia.gov" < /home/ikalash/Desktop/nightlyCDash/results_64bit
