#!/bin/bash

#Script to reconfigure Trilinos for linking against Dakota
#and then to build and install TriKota package

rm -rf $TRIBUILDDIR
mkdir $TRIBUILDDIR
cd $TRIBUILDDIR

#Configure Trilinos

echo "    Starting Trilinos cmake" ; date
if [ $MPI_BUILD ] ; then
  cp $ALBDIR/doc/nightlyTestHarness/shannon/do-cmake-trilinos-shannon-cuda .
  source $TRIBUILDDIR/do-cmake-trilinos-shannon-cuda > $TRILOUTDIR/trilinos_cuda_cmake.out 2>&1
fi

echo "    Finished Trilinos cmake, starting make" ; date

#Build Trilinos
/usr/bin/make -j 32  > $TRILOUTDIR/trilinos_cuda_make.out 2>&1
echo "    Finished Trilinos make, starting install" ; date
/usr/bin/make install > $TRILOUTDIR/trilinos_cuda_install.out 2>&1

