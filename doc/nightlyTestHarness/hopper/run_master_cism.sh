
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
    echo "ERROR: run_master: run_master.sh requires a file as an argument"
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
   echo; echo "... Performing $2 build of Albany and CISM"
   echo
   export MPI_BUILD=true
fi

#echo "... Deleting then Creating " $NIGHTLYDIR
#rm -rf $NIGHTLYDIR
#mkdir $NIGHTLYDIR

#-------------------------------------------
# Execute scripts for building trilinos, dakota, and albany
#-------------------------------------------

#echo; echo "...Starting Trilinos VOTD Checkout"
#time source $SCRIPTDIR/trilinos_checkout.sh

#echo; echo "...Starting Albany VOTD Checkout"
#time source $SCRIPTDIR/albany_checkout.sh

#echo; echo "...Starting Dakota VOTD wget and untar"
#time source $SCRIPTDIR/dakota_checkout.sh

#echo; echo "...Starting Trilinos full Build"
#time source $SCRIPTDIR/trilinos_build.sh

echo; echo "...Starting CISM felix_interface branch VOTD Checkout"
time source $SCRIPTDIR/cism_checkout.sh

echo; echo "...Starting Albany Build with CISM enabled"
time source $SCRIPTDIR/albany_build_cism.sh

echo; echo "...Starting CISM felix_interface branch Build"
time source $SCRIPTDIR/cism_build.sh 

#-------------------------------------------
# Execute albany tests
#-------------------------------------------
#echo; echo "...Starting Albany Tests"
#time source $SCRIPTDIR/albany_runtest.sh

#-------------------------------------------
# Execute parse output and send email scripts
#-------------------------------------------
# 
echo; echo "...Sending out email with results"
source $SCRIPTDIR/send_email_cism.sh
echo; echo "...Email sent!"
