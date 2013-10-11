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
  cp $ALBDIR/doc/hopper-albany-cmake .
  cp $ALBDIR/doc/hopper_modules.sh .
  source hopper_modules.sh > $ALBOUTDIR/albany_modules.out 2>&1
  source hopper-albany-cmake > $ALBOUTDIR/albany_cmake.out 2>&1
#else
#  cp $SCRIPTDIR/do-cmake-albany .
#  source ./do-cmake-albany > $ALBOUTDIR/albany_cmake.out 2>&1
fi

echo "    Finished Albany cmake, starting make" ; date

#/usr/bin/make -j 8 > $ALBOUTDIR/albany_make.out 2>&1
/usr/bin/make -j 4 > $ALBOUTDIR/albany_make.out 2>&1

echo "    Finished Albany make" ; date
