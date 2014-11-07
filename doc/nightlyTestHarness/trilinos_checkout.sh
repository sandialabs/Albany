#!/bin/bash

#-------------------------------------------
#  
# Prototype script to checkout, compile Trilinos
#
# This script is executed from run_master.sh
# 
# ToDo:
#        convert to Cmake
#
# BvBW  10/06/08
#
# AGS  04/09
#
# GAH  08/11
#-------------------------------------------

#-------------------------------------------
# setup and housekeeping
#-------------------------------------------

# Uncomment this for use stand-alone
#source $1

if [ -a $TRILDIR ]; then \rm -rf $TRILDIR
fi

if [ -a $TRILOUTDIR ]; then \rm -rf $TRILOUTDIR
fi


cd $NIGHTLYDIR
mkdir $TRILOUTDIR

#-------------------------------------------
# checkout Trilinos
#-------------------------------------------

#checks out master
git clone software.sandia.gov:/space/git/Trilinos > $TRILOUTDIR/trilinos_checkout.out 2>&1

cd $TRILDIR

echo; echo "   Starting SCOREC checkout..."
git clone git@github.com:SCOREC/core.git SCOREC > $TRILOUTDIR/scorec_checkout.out 2>&1
cd $TRILDIR/SCOREC
echo; echo "   ...finished SCOREC checkout."
