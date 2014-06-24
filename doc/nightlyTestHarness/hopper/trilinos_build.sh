

#!/bin/bash
#Script to reconfigure Trilinos for linking against Dakota
#and then to build and install TriKota package

rm -rf $TRIBUILDDIR
mkdir $TRIBUILDDIR
cd $TRIBUILDDIR

#Configure Trilinos

echo "    Starting Trilinos cmake" ; date
if [ $MPI_BUILD ] ; then
  cp $ALBDIR/doc/nightlyTestHarness/hopper/hopper-trilinos-cmake .
  cp $ALBDIR/doc/nightlyTestHarness/hopper/hopper_modules.sh .
  source hopper_modules.sh > $TRILOUTDIR/trilinos_modules.out 2>&1
  source hopper-trilinos-cmake > $TRILOUTDIR/trilinos_cmake.out 2>&1
#else
#  cp $SCRIPTDIR/do-cmake-trilinos .
#  source ./do-cmake-trilinos > $TRILOUTDIR/trilinos_cmake.out 2>&1
fi

echo "    Finished Trilinos cmake, starting make" ; date

#Build Trilinos
/usr/bin/make -j 4  > $TRILOUTDIR/trilinos_make.out 2>&1
echo "    Finished Trilinos make, starting install" ; date
/usr/bin/make install -j 4 > $TRILOUTDIR/trilinos_install.out 2>&1

# Get Dakota's boost out of the path
#rm -rf $TRILINSTALLDIR/include/boost
echo "    Finished Trilinos install" ; date
