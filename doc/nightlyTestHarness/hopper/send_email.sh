#!/bin/bash


TTT=`grep "Built target Albany" $ALBOUTDIR/albany_make.out`

/bin/mail -s "Albany ($ALBANY_BRANCH): $TTT" "albany-regression@software.sandia.gov" < $ALBOUTDIR/albany_make.out
