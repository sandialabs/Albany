#!/bin/bash

#Script to reconfigure Trilinos for linking against Dakota
#and then to build and install TriKota package

rm -rf $TRIBUILDDIRNOEPETRA
mkdir $TRIBUILDDIRNOEPETRA
cd $TRIBUILDDIRNOEPETRA

#Configure Trilinos

echo "    Starting Trilinos cmake" ; date
if [ $MPI_BUILD ] ; then
  cp $ALBDIR/doc/nightlyTestHarness/mockba/do-cmake-trilinos-mpi-mockba-no-epetra .
  source ./do-cmake-trilinos-mpi-mockba-no-epetra > $TRILOUTDIR/trilinos_cmake_no_epetra.out 2>&1
fi

echo "    Finished Trilinos cmake, starting make" ; date

#Build Trilinos
/usr/bin/make -j 8  > $TRILOUTDIR/trilinos_make_no_epetra.out 2>&1
echo "    Finished Trilinos make, starting install" ; date
/usr/bin/make install > $TRILOUTDIR/trilinos_install_no_epetra.out 2>&1

# Get Dakota's boost out of the path
rm -rf $TRILINSTALLDIRSPIRIT/include/boost
echo "    Finished Trilinos install" ; date
