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
     > $ALBOUTDIR/albany_runtests.out

ctest >> $ALBOUTDIR/albany_runtests.out

echo >> $ALBOUTDIR/albany_runtests.out
