#!/bin/bash


grep "Built" $CISMOUTDIR/cism_make.out > $CISMOUTDIR/cism_email.out

TTT=`grep "Built target simple_glide" $CISMOUTDIR/cism_email.out`

#/usr/bin/mail -s "Hopper Albany ($ALBANY_BRANCH): $TTT" "albany-regression@software.sandia.gov" < $ALBOUTDIR/albany_email.out
/usr/bin/mail -s "Hopper Glimmer/CISM (felix_interface): $TTT" "ikalashn@gmail.com" < $CISMOUTDIR/cism_email.out
#/bin/mail -s "Hopper Albany ($ALBANY_BRANCH): $TTT" "ikalash@sandia.gov" < $ALBOUTDIR/albany_email.out

