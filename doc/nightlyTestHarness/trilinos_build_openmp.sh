#!/bin/bash

#Script to reconfigure Trilinos for linking against Dakota
#and then to build and install TriKota package

rm -rf $TRIBUILDDIROPENMP
mkdir $TRIBUILDDIROPENMP
cd $TRIBUILDDIROPENMP

#Configure Trilinos

echo "    Starting Trilinos cmake" ; date
if [ $MPI_BUILD ] ; then
  cp $ALBDIR/doc/nightlyTestHarness/do-cmake-trilinos-mpi-tpetra-openmp .
  source ./do-cmake-trilinos-mpi-tpetra-openmp > $TRILOUTDIR/trilinos_openmp_cmake.out 2>&1
else
  cp $ALBDIR/doc/nightlyTestHarness/do-cmake-trilinos-tpetra .
  source ./do-cmake-trilinos-tpetra-no-scorec > $TRILOUTDIR/trilinos_openmp_cmake.out 2>&1
fi

echo "    Finished Trilinos cmake, starting make" ; date

#Build Trilinos
/usr/bin/make -j 32  > $TRILOUTDIR/trilinos_openmp_make.out 2>&1
echo "    Finished Trilinos make, starting install" ; date
/usr/bin/make install > $TRILOUTDIR/trilinos_openmp_install.out 2>&1

# Get Dakota's boost out of the path
rm -rf $TRILINSTALLDIROPENMP/include/boost
echo "    Finished Trilinos install" ; date
