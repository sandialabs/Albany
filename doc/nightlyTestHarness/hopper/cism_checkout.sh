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

if [ -a $NIGHTLYDIR/felix_interface ]; then \rm -rf $NIGHTLYDIR/felix_interface
fi

if [ -a $CISMOUTDIR ]; then \rm -rf $CISMOUTDIR
fi

cd $NIGHTLYDIR
mkdir $CISMOUTDIR

#-------------------------------------------
# git clone Albany
#-------------------------------------------

echo "     Checking out CISM felix_interface branch "
module load subversion
svn checkout http://oceans11.lanl.gov/svn/PISCEES/branches/felix_interface > $CISMOUTDIR/cism_checkout.out 2>&1
echo "     Finished checkout out CISM felix_interface branch "

