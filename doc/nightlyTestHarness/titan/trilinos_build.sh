#!/bin/bash
#Script to reconfigure Trilinos for linking against Dakota
#and then to build and install TriKota package

rm -rf $TRIBUILDDIR
mkdir $TRIBUILDDIR
cd $TRIBUILDDIR

#Configure Trilinos

echo "    Starting Trilinos cmake" ; date
cp $SCRIPTDIR/titan-trilinos-cmake .
cp $SCRIPTDIR/titan_modules.sh .
source titan_modules.sh > $TRILOUTDIR/trilinos_modules.out 2>&1
source titan-trilinos-cmake > $TRILOUTDIR/trilinos_cmake.out 2>&1

echo "    Finished Trilinos cmake, starting make" ; date

#Build Trilinos
/usr/bin/make -j 16  > $TRILOUTDIR/trilinos_make.out 2>&1
echo "    Finished Trilinos make, starting install" ; date
/usr/bin/make install > $TRILOUTDIR/trilinos_install.out 2>&1

# Get Dakota's boost out of the path
#rm -rf $TRILINSTALLDIR/include/boost
echo "    Finished Trilinos install" ; date
