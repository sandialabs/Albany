#!/bin/bash


grep "Built" $ALBOUTDIR/albany_make.out > $ALBOUTDIR/albany_email.out

TTT=`grep "Built target Albany" $ALBOUTDIR/albany_email.out`

/usr/bin/mail -s "Hopper Albany ($ALBANY_BRANCH): $TTT" "albany-regression@software.sandia.gov" -f "ikalash@nersc.gov" < $ALBOUTDIR/albany_email.out
/usr/bin/mail -s "Hopper Albany ($ALBANY_BRANCH): $TTT" "ikalashn@gmail.com" -f "ikalash@nersc.gov" < $ALBOUTDIR/albany_email.out

