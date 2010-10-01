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

eg clone software.sandia.gov:/space/git/Trilinos > $TRILOUTDIR/trilinos_checkout.out 2>&1
cd Trilinos

echo "BRANCH 10_6"
eg switch trilinos-release-10-6-branch

eg clone software.sandia.gov:/space/git/preCopyrightTrilinos > $TRILOUTDIR/trilinos_preC_checkout.out 2>&1
cd ..

