#!/bin/bash

SCRIPTDIR=/home/ikalash/nightlyAlbanyTests/
NIGHTLYDIR=$SCRIPTDIR/Results
ALBOUTDIR=$NIGHTLYDIR/Albany_out
ALBANY_BRANCH=DynRankViewIntrepid2Refactor

grep "Built" $ALBOUTDIR/albany_make.out >& $ALBOUTDIR/albany_email.out

mail -s "Shannon Albany ($ALBANY_BRANCH) build" "ikalash@sandia.gov" < $ALBOUTDIR/albany_email.out

