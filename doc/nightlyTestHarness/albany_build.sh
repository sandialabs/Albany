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
# svn checkout Albany
#-------------------------------------------

svn export svn+ssh://software.sandia.gov/space/sandiasvn/private/DemoApps/trunk/Albany Albany > $ALBOUTDIR/albany_checkout.out 2>&1

#-------------------------------------------
# autoconf:  configure and make Albany
#-------------------------------------------

cd $ALBDIR
rm -rf $ALBDIR/LINUX_DEBUG
mkdir $ALBDIR/LINUX_DEBUG
cd $ALBDIR/LINUX_DEBUG
../configure --with-gnumake --with-trilinos="$TRILINSTALLDIR" \
  --with-dakota="$TRILINSTALLDIR" > $ALBOUTDIR/albany_configure.out 2>&1

/usr/bin/make -j 4 > $ALBOUTDIR/albany_make.out 2>&1


#-------------------------------------------
# cmake:  configure and make Albany 
#-------------------------------------------

cd $ALBDIR
rm -rf $ALBDIR/CMAKE_BUILD
mkdir $ALBDIR/CMAKE_BUILD
cd $ALBDIR/CMAKE_BUILD
cmake \
 -D ALBANY_TRILINOS_DIR:FILEPATH="$TRILINSTALLDIR" \
 -D CMAKE_VERBOSE_MAKEFILE:BOOL=ON \
 ..   > $ALBOUTDIR/albany_cmake.out 2>&1

/usr/bin/make -j 4 > $ALBOUTDIR/albany_cmake_make.out 2>&1
