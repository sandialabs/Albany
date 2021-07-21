#!/bin/bash

#-------------------------------------------
#  
# Prototype script to checkout, compile Trilinos
#
# This script is executed from run_tpetra.sh
# 
# ToDo:
#        convert to Cmake
#
# BvBW  10/06/08
#
# AGS  04/09
#
# GAH  08/11
#-------------------------------------------

#-------------------------------------------
# setup and housekeeping
#-------------------------------------------

# Uncomment this for use stand-alone
#source $1

if [ -a $TRILDIR ]; then \rm -rf $TRILDIR
fi

if [ -a $TRILOUTDIR ]; then \rm -rf $TRILOUTDIR
fi


cd $NIGHTLYDIR
mkdir $TRILOUTDIR

#-------------------------------------------
# checkout Trilinos
#-------------------------------------------

#checks out master
git clone git@github.com:trilinos/Trilinos.git > $TRILOUTDIR/trilinos_checkout.out 2>&1
cd $TRILDIR
git checkout $TRILINOS_BRANCH

#check out Dakota
#echo; echo "   Starting Dakota checkout..."
#cd $TRILDIR/packages/TriKota
export https_proxy="https://proxy.ca.sandia.gov:80"
export http_proxy="http://proxy.ca.sandia.gov:80"
#wget -nv --no-check-certificate https://dakota.sandia.gov/sites/default/files/distributions/public/dakota-6.9-release-public.src.tar.gz -v
#tar -zxvf dakota-6.9-release-public.src.tar.gz
#rm -rf dakota-6.9-release-public.src.tar.gz
#mv dakota-6.9.0.src Dakota
#apply patch - see Trilinos issue #4771
#sed -i '70i #include <cmath>' Dakota/packages/external/JEGA/Utilities/src/DiscreteDesignVariableNature.cpp
#echo; echo "   ...finished Dakota checkout."
#echo; echo "   Copying Dakota directory into Trilinos..."
#cp -r /nightlyAlbanyTests/Dakota .
#echo; echo "   ...finished Dakota copy into Trilinos."

#echo; echo "   Copying DTK directory into Trilinos..."
echo; echo "   Starting DTK checkout..."
cd $TRILDIR
#cp -r /home/ikalash/nightlyAlbanyTests/DataTransferKit-2.0.0 DataTransferKit 
git clone git@github.com:ikalash/DataTransferKit.git
cd DataTransferKit
git checkout dtk-2.0-tpetra-static-graph
#git clone git@github.com:ORNL-CEES/DTKData.git
#echo; echo "   ...finished DTK copy into Trilinos."
echo; echo "   ...finished DTK checkout."

#echo; echo "   Starting SCOREC checkout..."
#cd $TRILDIR
#git clone git@github.com:SCOREC/core.git SCOREC > $TRILOUTDIR/scorec_checkout.out 2>&1
#echo; echo "   ...finished SCOREC checkout."
