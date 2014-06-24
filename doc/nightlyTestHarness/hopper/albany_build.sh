#!/bin/bash


#-------------------------------------------
# cmake:  configure and make Albany 
#-------------------------------------------

cd $ALBDIR
rm -rf $ALBDIR/build
mkdir $ALBDIR/build
cd $ALBDIR/build

echo "    Starting Albany cmake" ; date

if [ $MPI_BUILD ] ; then
  cp $ALBDIR/doc/nightlyTestHarness/hopper/hopper-albany-cmake .
  cp $ALBDIR/doc/nightlyTestHarness/hopper/hopper_modules.sh .
  source hopper_modules.sh > $ALBOUTDIR/albany_modules.out 2>&1
  source hopper-albany-cmake > $ALBOUTDIR/albany_cmake.out 2>&1
fi

echo "    Finished Albany cmake, starting make" ; date

/usr/bin/make -j 4 Albany > $ALBOUTDIR/albany_make.out 2>&1

echo "    Finished Albany make" ; date
