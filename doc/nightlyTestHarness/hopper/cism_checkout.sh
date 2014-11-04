#!/bin/bash

#-------------------------------------------
#  
# Prototype script to checkout, compile 
# Albany 
# 
# This script is executed from run_master.sh
#
# BvBW  10/06/08
# AGS  04/09
#-------------------------------------------

#-------------------------------------------
# setup and housekeeping
#-------------------------------------------

if [ -a $NIGHTLYDIR/cism-piscees ]; then \rm -rf $NIGHTLYDIR/cism_piscees
fi

if [ -a $CISMOUTDIR ]; then \rm -rf $CISMOUTDIR
fi

cd $NIGHTLYDIR
mkdir $CISMOUTDIR

#-------------------------------------------
# git clone Albany
#-------------------------------------------

echo "     Checking out CISM "
git clone git@github.com:ACME-Climate/cism-piscees.git > $CISMOUTDIR/cism_checkout.out 2>&1
echo "     Finished checkout out CISM "

echo "Switching CISM to branch ", $CISM_BRANCH
cd cism-piscees
git checkout $CISM_BRANCH

