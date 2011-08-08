#!/bin/bash

TTT=`grep "tests failed" $ALBOUTDIR/albany_runtests.out`

/bin/mail -s "Albany ($ALBANY_BRANCH): $TTT" "albany-regression@software.sandia.gov" < $ALBOUTDIR/albany_runtests.out
