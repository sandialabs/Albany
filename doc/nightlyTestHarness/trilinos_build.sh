#!/bin/bash

#Script to reconfigure Trilinos for linking against Dakota
#and then to build and install TriKota package

mkdir $TRILINSTALLDIR
cd $TRILINSTALLDIR

cp $NIGHTLYDIR/do-cmake-trilinos .

#Configure Trilinos
echo "    Starting Trilinos cmake" ; date
source ./do-cmake-trilinos > $TRILOUTDIR/trilinos_cmake.out 2>&1
echo "    Finished Trilinos cmake, starting make" ; date

#Build Trilinos
/usr/bin/make -j 4  > $TRILOUTDIR/trilinos_make.out 2>&1
echo "    Finished Trilinos make, starting install" ; date
/usr/bin/make install > $TRILOUTDIR/trilinos_install.out 2>&1

# Get Dakota's boost out of the path
rm -r $TRILINSTALLDIR/include/boost
echo "    Finished Trilinos install" ; date
