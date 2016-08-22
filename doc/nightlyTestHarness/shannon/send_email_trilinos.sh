#!/bin/bash

SCRIPTDIR=/home/ikalash/nightlyAlbanyTests/
NIGHTLYDIR=$SCRIPTDIR/Results
TRILOUTDIR=$NIGHTLYDIR/Trilinos_out
TRILINOS_BRANCH=develop

grep "Built" $TRILOUTDIR/trilinos_cuda_make.out >& $TRILOUTDIR/trilinos_email.out
sed -i -e 's,\[,Q,g'  $TRILOUTDIR/trilinos_email.out
sed -i -e 's,Q,\n[,g' $TRILOUTDIR/trilinos_email.out 
sed -i -e 's,\n,\t,g' $TRILOUTDIR/trilinos_email.out 

mail -s "Shannon Trilinos ($TRILINOS_BRANCH) build" "ikalash@sandia.gov" < $TRILOUTDIR/trilinos_email.out

