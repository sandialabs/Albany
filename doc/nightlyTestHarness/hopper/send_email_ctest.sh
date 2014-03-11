#!/bin/bash

source $1

TTT=`grep "tests failed" $ALBOUTDIR/albany_runtests.out`

/usr/bin/mail -s "Hopper Albany ($ALBANY_BRANCH): $TTT" "albany-regression@software.sandia.gov" -f "ikalash@nersc.gov" < $ALBOUTDIR/albany_runtests.out
/usr/bin/mail -s "Hopper Albany ($ALBANY_BRANCH): $TTT" "ikalashn@gmail.com" -f "ikalash@nersc.gov" < $ALBOUTDIR/albany_runtests.out

