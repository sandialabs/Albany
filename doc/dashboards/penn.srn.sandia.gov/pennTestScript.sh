#!/bin/bash

source /home/agsalin/.bash_profile

# Set these to whatever you like
SCRIPTDIR=/home/agsalin/pennTests
NIGHTLYDIR=/home/agsalin/pennTest/Results

#set this correctly
BOOSTDIR=/usr/local
NETCDFDIR=/home/agsalin/install/netcdf-4.2/install
HDF5DIR=/home/agsalin/install/hdf5-1.8.11/hdf5/
#set this correctly, to software.sandia.gov user name
SSGUSER=agsalin

# set to OFF to skip Dakota build
## NOT WORKING YET
#DAKOTABOOL="ON"

# Don't change these
TRILDIR=$NIGHTLYDIR/Trilinos
TRILBUILDDIR=$NIGHTLYDIR/Trilinos/build
TRILINSTALLDIR=$NIGHTLYDIR/Trilinos/build/install
ALBDIR=$NIGHTLYDIR/Albany
TRIKOTADIR=$TRILDIR/packages/TriKota

# Set these to whatever you like
TRILOUTDIR=$NIGHTLYDIR/Trilinos_out
ALBOUTDIR=$NIGHTLYDIR/Albany_out
DAKOUTDIR=$NIGHTLYDIR/Dakota_out

# Must be set: master is typical
TRILINOS_BRANCH=master
ALBANY_BRANCH=master

#-------------------------------------------


echo "...Deleting then Creating " $NIGHTLYDIR
rm -rf $NIGHTLYDIR
mkdir $NIGHTLYDIR

cp $SCRIPTDIR/do-cmake-trilinos $NIGHTLYDIR/.

#-------------------------------------------
# Execute scripts for building trilinos, dakota, and albany
#-------------------------------------------

#-------------------------------------------
# checkout Trilinos
#-------------------------------------------

echo; echo "...Starting Trilinos VOTD Checkout"
if [ -a $TRILDIR ]; then \rm -rf $TRILDIR
fi
if [ -a $TRILOUTDIR ]; then \rm -rf $TRILOUTDIR
fi
cd $NIGHTLYDIR
mkdir $TRILOUTDIR

git clone software.sandia.gov:/space/git/Trilinos > $TRILOUTDIR/trilinos_checkout.out 2>&1

#-------------------------------------------
# checkout Dakota
#-------------------------------------------
echo; echo "...Starting Dakota VOTD wget and untar"

if [ -a $NIGHTLYDIR/dakota-stable.src.tar ]; then \rm $NIGHTLYDIR/dakota-stable.src.tar
fi
if [ -a $NIGHTLYDIR/dakota-stable.src.tar.gz ]; then \rm $NIGHTLYDIR/dakota-stable.src.tar.gz
fi

if [ -a $DAKOUTDIR ]; then \rm -rf $DAKOUTDIR
fi
mkdir  $DAKOUTDIR
cd $NIGHTLYDIR

wget -nv --no-check-certificate https://dakota.sandia.gov/sites/default/files/distributions/public/dakota-6.2-public.src.tar.gz >  $DAKOUTDIR/dakota_wget.out 
cd $TRIKOTADIR
tar xvfz $NIGHTLYDIR/dakota-6.2-public.src.tar.gz  > $DAKOUTDIR/dakota_untar.out 2>&1
mv dakota-6.2.0.src Dakota

#-------------------------------------------
# Build and install Trilinos
#-------------------------------------------
echo; echo "...Starting full Trilinos Build"
rm -rf $TRILBUILDDIR
mkdir $TRILBUILDDIR
cd $TRILBUILDDIR

#Configure Trilinos
echo "    Starting Trilinos cmake" ; date
cp $SCRIPTDIR/do-cmake-trilinos .
source ./do-cmake-trilinos > $TRILOUTDIR/trilinos_cmake.out 2>&1
echo "    Finished Trilinos cmake, starting make" ; date

#Build Trilinos
/usr/bin/make -j 20  > $TRILOUTDIR/trilinos_make.out 2>&1
echo "    Finished Trilinos make, starting install" ; date
/usr/bin/make install > $TRILOUTDIR/trilinos_install.out 2>&1
echo "    Finished Trilinos install" ; date

#-------------------------------------------
# Execute albany tests
#-------------------------------------------
echo; echo "...Starting Albany VOTD clone and Build"

if [ -a $NIGHTLYDIR/Albany ]; then \rm -rf $NIGHTLYDIR/Albany
fi
if [ -a $ALBOUTDIR ]; then \rm -rf $ALBOUTDIR
fi
cd $NIGHTLYDIR
mkdir $ALBOUTDIR

git clone git@github.com:gahansen/Albany > $ALBOUTDIR/albany_checkout.out 2>&1

cd Albany
echo "Switching Albany to branch ", $ALBANY_BRANCH
git checkout $ALBANY_BRANCH

# cmake:  configure and make Albany 

cd $ALBDIR
rm -rf $ALBDIR/build
mkdir $ALBDIR/build
cd $ALBDIR/build

echo "    Starting Albany cmake" ; date
cp $SCRIPTDIR/do-cmake-albany .
source ./do-cmake-albany > $ALBOUTDIR/albany_cmake.out 2>&1
echo "    Finished Albany cmake, starting make" ; date

echo; echo "...Starting Albany Build and Test"

cd $ALBDIR/build
echo "------------------CTEST----------------------" \
     > $ALBOUTDIR/albany_ctest.out

# This re-configures based on CMakeCache, builds, tests, posts
ctest -D Experimental >> $ALBOUTDIR/albany_ctest.out

#-------------------------------------------
# Execute parse output and send email scripts
#-------------------------------------------
# 

TTT=`grep "tests failed" $ALBOUTDIR/albany_ctest.out`

/bin/mail -s "Albany ($ALBANY_BRANCH): $TTT" "agsalin@sandia.gov" < $ALBOUTDIR/albany_ctest.out
