#!/bin/bash


#-------------------------------------------
# cmake:  configure and make Albany 
#-------------------------------------------

cd $CISMDIR/builds/hopper-gnu-felix

echo "    Starting CISM cmake" ; date

if [ $MPI_BUILD ] ; then
  cp $ALBDIR/doc/nightlyTestHarness/hopper/hopper-cism-albany-cmake .
  cp $ALBDIR/doc/nightlyTestHarness/hopper/hopper_modules.sh .
  source hopper_modules.sh > $CISMOUTDIR/cism_modules.out 2>&1
  source hopper-cism-albany-cmake > $CISMOUTDIR/cism_cmake.out 2>&1
fi

echo "    Finished CISM cmake, starting make" ; date

/usr/bin/make -j 8 > $CISMOUTDIR/cism_make.out 2>&1

echo "    Finished CISM make" ; date
