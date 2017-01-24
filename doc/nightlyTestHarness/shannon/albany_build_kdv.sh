#!/bin/bash

#-------------------------------------------
#  
# Prototype script to checkout, compile 
# Albany 
# 
# This script is executed from run_master_tpetra.sh
#
# BvBW  10/06/08
# AGS  04/09
#-------------------------------------------

#-------------------------------------------
# cmake:  configure and make Albany 
#-------------------------------------------

cd $ALBDIR
rm -rf $ALBDIR/build-cuda
mkdir $ALBDIR/build-cuda
cd $ALBDIR/build-cuda

echo "    Starting Albany cmake" ; date

if [ $MPI_BUILD ] ; then
  cp $ALBDIR/doc/nightlyTestHarness/shannon/do-cmake-albany-shannon-cuda .
  source $ALBDIR/build-cuda/do-cmake-albany-shannon-cuda > $ALBOUTDIR/albany_cmake.out 2>&1
fi

echo "    Finished Albany cmake, starting make" ; date

#/usr/bin/make -j 8 > $ALBOUTDIR/albany_make.out 2>&1
/usr/bin/make -j 8 > $ALBOUTDIR/albany_make.out 2>&1
echo "    Finished Albany make!" ; date

