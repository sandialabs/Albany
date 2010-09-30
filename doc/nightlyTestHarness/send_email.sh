#!/bin/bash

TTT=`grep "tests failed" $ALBOUTDIR/albany_runtests.out`

/bin/mail -s "Albany: $TTT" "DemoApps-regression@software.sandia.gov" < $ALBOUTDIR/albany_runtests.out
