#!/bin/bash


grep "Built" $CISMOUTDIR/cism_make.out > $CISMOUTDIR/cism_email.out

TTT=`grep "Built target simple_glide" $CISMOUTDIR/cism_email.out`

/usr/bin/mail -s "Hopper CISM-Albany: $TTT" "ikalashn@gmail.com" -f "ikalash@nersc.gov" < $CISMOUTDIR/cism_email.out
/usr/bin/mail -s "Hopper CISM-Albany: $TTT" "ikalash@sandia.gov, agsalin@sandia.gov" -f "ikalash@nersc.gov" < $CISMOUTDIR/cism_email.out

