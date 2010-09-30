#!/bin/bash

#-------------------------------------------
#  
# Prototype script to checkout, compile Trilinos
# Albany, Dakota, in addition to performing
# "runtests" for Albany
# 
# This scripts calls various subscripts
#
# ToDo: 
#        convert to Cmake
#
# BvBW  10/06/08
#
# AGS  04/09
#-------------------------------------------

#-------------------------------------------
# Get paths as environment variabls from input file $1
#-------------------------------------------

if [ ! $1 ] ; then
    echo "ERROR: run_master: run_master.sh requires a file as an argumnet"
    echo "You must define env variables with required paths!"
    exit
fi

if [ -s $1 ] ; then
  echo "Sourcing Environment variable file for required paths: " $1
  source $1
else 
  echo "ERROR: run_master: File not found: $1 argument = " $1
  echo "You must define env variables with required paths!"
  exit
fi

echo "...Deleting then Creating " $NIGHTLYDIR
rm -rf $NIGHTLYDIR
mkdir $NIGHTLYDIR

cp $SCRIPTDIR/do-cmake-trilinos $NIGHTLYDIR/.

#-------------------------------------------
# Execute scripts for building trilinos, dakota, and albany
#-------------------------------------------

echo; echo "...Starting Trilinos VOTD Checkout"
time source $SCRIPTDIR/trilinos_checkout.sh
echo; echo "...Starting Dakota VOTD wget and untar"
time source $SCRIPTDIR/dakota_checkout.sh
echo; echo "...Starting Trilinos full Build, including Trikota"
time source $SCRIPTDIR/trilinos_build_all.sh
echo; echo "...Starting Albany VOTD Export and Build"
time source $SCRIPTDIR/albany_build.sh

#-------------------------------------------
# Execute albany tests
#-------------------------------------------

echo; echo "...Starting Albany Tests"
time source $SCRIPTDIR/albany_runtest.sh

#-------------------------------------------
# Execute parse output and send email scripts
#-------------------------------------------
# 
source $SCRIPTDIR/send_email.sh
