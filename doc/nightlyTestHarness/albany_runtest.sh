#!/bin/bash

#-------------------------------------------
#  
# Prototype script to execute regression
# tests
#
# This script is executed from run_master.sh
#
# BvBW  10/06/08
# AGS  04/09
#-------------------------------------------

#-------------------------------------------
# setup and housekeeping
#-------------------------------------------

cd $NIGHTLYDIR

if [ ! -d $ALBOUTDIR ]; then mkdir $ALBOUTDIR
fi

#-------------------------------------------
# run tests in Albany
#-------------------------------------------

cd $ALBDIR/CMAKE_BUILD
echo "------------------CTEST----------------------" \
     >> $ALBOUTDIR/albany_runtests.out

ctest >> $ALBOUTDIR/albany_runtests.out

echo >> $ALBOUTDIR/albany_runtests.out

# Repeat for autoconf build
echo "------------------runtests----------------------" \
     >> $ALBOUTDIR/albany_runtests.out

cd $ALBDIR/test/utilities
./runtestsDakotaSGAnalysis --tramonto-dir=$NIGHTLYDIR/Albany \
                   --build-dir=LINUX_DEBUG --category=default \
                   --verbosity=2 >> $ALBOUTDIR/albany_runtests.out


