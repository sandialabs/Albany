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

eg clone software.sandia.gov:/space/git/Albany > $ALBOUTDIR/albany_checkout.out 2>&1
cd Albany
echo "Switching Albany to branch ", $ALBANY_BRANCH
eg switch $ALBANY_BRANCH

#-------------------------------------------
# cmake:  configure and make Albany 
#-------------------------------------------

cd $ALBDIR
rm -rf $ALBDIR/LINUX_DEBUG
mkdir $ALBDIR/LINUX_DEBUG
cd $ALBDIR/LINUX_DEBUG
cmake \
 -D ALBANY_TRILINOS_DIR:FILEPATH="$TRILINSTALLDIR" \
 -D CMAKE_VERBOSE_MAKEFILE:BOOL=ON \
 -D ENABLE_LCM:BOOL=ON \
 ..   > $ALBOUTDIR/albany_cmake.out 2>&1

/usr/bin/make -j 4 > $ALBOUTDIR/albany_make.out 2>&1
