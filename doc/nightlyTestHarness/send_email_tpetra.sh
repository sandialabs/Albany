#!/bin/bash

source $1 

TTT=`grep "tests failed" $ALBOUTDIR/albany_runtests.out`

#/bin/mail -s "Albany ($ALBANY_BRANCH): $TTT" "albany-regression@software.sandia.gov" < $ALBOUTDIR/albany_runtests.out
/bin/mail -s "Albany ($ALBANY_BRANCH): $TTT" "ikalash@sandia.gov" < $ALBOUTDIR/albany_runtests.out
/bin/mail -s "Albany ($ALBANY_BRANCH): $TTT" "agsalin@sandia.gov" < $ALBOUTDIR/albany_runtests.out
/bin/mail -s "Albany ($ALBANY_BRANCH): $TTT" "jcortia@sandia.gov" < $ALBOUTDIR/albany_runtests.out
/bin/mail -s "Albany ($ALBANY_BRANCH): $TTT" "gahanse@sandia.gov" < $ALBOUTDIR/albany_runtests.out
