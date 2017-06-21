#!/bin/bash

#-------------------------------------------
#  
# Prototype script to checkout, compile Trilinos
#
# This script is executed from run_master.sh
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
echo; echo "   Starting Dakota checkout..."
cd $TRILDIR/packages/TriKota
export https_proxy="https://wwwproxy.ca.sandia.gov:80"
export http_proxy="http://wwwproxy.ca.sandia.gov:80"
wget -nv --no-check-certificate https://dakota.sandia.gov/sites/default/files/distributions/public/dakota-6.5-public.src.tar.gz -v
tar -zxvf dakota-6.5-public.src.tar.gz 
rm -rf dakota-6.5-public.src.tar.gz 
mv dakota-6.5.0.src Dakota
echo; echo "   ...finished Dakota checkout."

echo; echo "   Copying DTK directory into Trilinos..."
cd $TRILDIR
cp -r /home/ikalash/nightlyAlbanyTests/DataTransferKit-2.0.0 DataTransferKit 
#git clone git@github.com:ORNL-CEES/DataTransferKit.git
#cd DataTransferKit
#git clone git@github.com:ORNL-CEES/DTKData.git
echo; echo "   ...finished DTK copy into Trilinos."



#echo; echo "   Starting SCOREC checkout..."
#git clone git@github.com:SCOREC/core.git SCOREC > $TRILOUTDIR/scorec_checkout.out 2>&1
#cd $TRILDIR/SCOREC
#echo; echo "   ...finished SCOREC checkout."
