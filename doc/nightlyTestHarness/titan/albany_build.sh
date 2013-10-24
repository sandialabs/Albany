#!/bin/bash


#-------------------------------------------
# cmake:  configure and make Albany 
#-------------------------------------------

cd $ALBDIR
rm -rf $ALBDIR/build
mkdir $ALBDIR/build
cd $ALBDIR/build

echo "    Starting Albany cmake" ; date

cp $SCRIPTDIR/titan-albany-cmake .
cp $SCRIPTDIR/titan_modules.sh .
source titan_modules.sh > $ALBOUTDIR/albany_modules.out 2>&1
source titan-albany-cmake > $ALBOUTDIR/albany_cmake.out 2>&1

echo "    Finished Albany cmake, starting make" ; date

/usr/bin/make -j 16 Albany > $ALBOUTDIR/albany_make.out 2>&1

echo "    Finished Albany make" ; date
