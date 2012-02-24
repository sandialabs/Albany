#!/bin/bash -l

# This script will build the Albany webpage
# There are implicit requirements that the machine you intend 
# to view the webpage from have PHP running, and that doxygen and dot
# are installed.

# set the target location (where you want to view to webpage from)
TARGET_DIR=~/Sites

# set the location of your Albany repository
ALBANY_DIR=/scratch/LCM/Web

##
## The following will run doxygen and copy the relevant stuff over to the target location
##

cd $ALBANY_DIR/doc/doxygen
doxygen
cp -r $ALBANY_DIR/doc/webpage $TARGET_DIR
cp -r $ALBANY_DIR/doc/doxygen $TARGET_DIR/webpage
cd $TARGET_DIR
chmod -R a+x webpage