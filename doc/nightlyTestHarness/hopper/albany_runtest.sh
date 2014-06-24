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
# run tests in Albany - FELIX only for now
#-------------------------------------------

cd $ALBDIR/build/examples/FELIX_Stokes

cp $ALBDIR/doc/nightlyTestHarness/hopper/hopper_modules.sh .
cp $ALBDIR/doc/nightlyTestHarness/hopper/job.pbs .
qsub job.pbs 

