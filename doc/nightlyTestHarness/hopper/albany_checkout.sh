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

if [ -a $NIGHTLYDIR/Albany ]; then \rm -rf $NIGHTLYDIR/Albany
fi

if [ -a $ALBOUTDIR ]; then \rm -rf $ALBOUTDIR
fi

cd $NIGHTLYDIR
mkdir $ALBOUTDIR

#-------------------------------------------
# git clone Albany
#-------------------------------------------

git clone git@github.com:gahansen/Albany.git > $ALBOUTDIR/albany_checkout.out 2>&1

cd Albany
echo "Switching Albany to branch ", $ALBANY_BRANCH
git checkout $ALBANY_BRANCH

