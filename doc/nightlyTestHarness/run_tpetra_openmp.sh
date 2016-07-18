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
#
# GAH  08/11
#-------------------------------------------

# Bail out if we have an unset variable
#set -o nounset

# Bail out on error
set -o errexit

#-------------------------------------------
# Get paths as environment variabls from input file $1
#-------------------------------------------

if [ ! $1 ] ; then
    echo "ERROR: run_master: run_master_tpetra.sh requires a file as an argument"
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

if [ "$2" = "MPI" ] ; then
   echo; echo "... Performing $2 build of Albany and Trilinos"
   echo
   export MPI_BUILD=true
fi

#echo "... Deleting then Creating " $NIGHTLYDIR
#rm -rf $NIGHTLYDIR
#mkdir $NIGHTLYDIR

#-------------------------------------------
# Execute scripts for building trilinos, dakota, and albany
#-------------------------------------------

#echo; echo "...Sourcing bashrc"
#time source /home/ikalash/.bashrc

#echo; echo "...Starting Trilinos VOTD Checkout"
#time source $SCRIPTDIR/trilinos_checkout_openmp.sh

#echo; echo "...Starting Albany VOTD Checkout"
#time source $SCRIPTDIR/albany_checkout_tpetra.sh

#echo; echo "...Starting Dakota VOTD wget and untar"
#time source $SCRIPTDIR/dakota_checkout.sh

echo; echo "...Starting Trilinos full Build"
time source $SCRIPTDIR/trilinos_build_openmp.sh

rm -rf $NIGHTLYDIR/Albany 

#echo; echo "...Starting Albany Build (Albany and AlbanyT)"
#time source $SCRIPTDIR/albany_build_tpetra.sh

#-------------------------------------------
# Execute albany tests
#-------------------------------------------
#echo; echo "...Starting Albany Tests (Albany and AlbanyT)"
#time source $SCRIPTDIR/albany_runtest_tpetra.sh

#-------------------------------------------
# Execute scripts for building trilinos, dakota, and albany
#-------------------------------------------
#echo; echo "...Starting Albany Build (AlbanyT only)"
#time source $SCRIPTDIR/albany_build_tpetra_albanyTonly.sh

#-------------------------------------------
# Execute albany tests
#-------------------------------------------
#echo; echo "...Starting Albany Tests (AlbanyT only)"
#time source $SCRIPTDIR/albany_runtest_tpetra_albanyTonly.sh

#-------------------------------------------
# Execute scripts for building trilinos, dakota, and albany

#-------------------------------------------
# Execute parse output and send email scripts
#-------------------------------------------
# 
#source $SCRIPTDIR/send_email_hack.sh
