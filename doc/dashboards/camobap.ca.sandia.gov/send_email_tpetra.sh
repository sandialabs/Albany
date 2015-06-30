#!/bin/bash

#source $1 

TTT=`grep "tests failed" /home/ikalash/Desktop/nightlyCDash/nightly_log.txt`

#/bin/mail -s "Albany ($ALBANY_BRANCH): $TTT" "albany-regression@software.sandia.gov" < $ALBOUTDIR/albany_runtests.out
/bin/mail -s "Albany (master, AlbanyT only): $TTT" "ikalash@sandia.gov" < /home/ikalash/Desktop/nightlyCDash/results
/bin/mail -s "Albany (master, AlbanyT only): $TTT" "agsalin@sandia.gov" < /home/ikalash/Desktop/nightlyCDash/results
#/bin/mail -s "Albany ($ALBANY_BRANCH): $TTT" "jcortia@sandia.gov" < $ALBOUTDIR/albany_runtests.out
/bin/mail -s "Albany (master, AlbanyT only): $TTT" "gahanse@sandia.gov" < /home/ikalash/Desktop/nightlyCDash/results
/bin/mail -s "Albany (master, AlbanyT only): $TTT" "ambradl@sandia.gov" < /home/ikalash/Desktop/nightlyCDash/results
#/bin/mail -s "Albany ($ALBANY_BRANCH): $TTT" "mperego@sandia.gov" < $ALBOUTDIR/albany_runtests.out
#/bin/mail -s "Albany ($ALBANY_BRANCH): $TTT" "ikalashn@gmail.com" < $ALBOUTDIR/albany_runtests.out
