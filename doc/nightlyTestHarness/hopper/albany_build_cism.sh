#!/bin/bash


#-------------------------------------------
# cmake:  configure and make Albany 
#-------------------------------------------

cd $ALBDIR
if [ -a $ALBCISMOUTDIR ]; then \rm -rf $ALBCISMOUTDIR
fi
mkdir $ALBCISMOUTDIR
rm -rf $ALBDIR/build_cism
mkdir $ALBDIR/build_cism
cd $ALBDIR/build_cism


echo "    Starting Albany cmake" ; date

if [ $MPI_BUILD ] ; then
  cp $ALBDIR/doc/nightlyTestHarness/hopper/hopper-albany-cism-cmake .
  cp $ALBDIR/doc/nightlyTestHarness/hopper/hopper_modules.sh .
  source hopper_modules.sh > $ALBCISMOUTDIR/albany_modules.out 2>&1
  source hopper-albany-cism-cmake > $ALBCISMOUTDIR/albany_cmake.out 2>&1
fi

echo "    Finished Albany cmake, starting make" ; date

/usr/bin/make -j 4 Albany > $ALBCISMOUTDIR/albany_make.out 2>&1

echo "    Finished Albany make" ; date

echo "    Starting Albany make install" ; date
/usr/bin/make install -j 4 > $ALBCISMOUTDIR/albany_install.out 2>&1
echo "    Finished Albany make install" ; date
