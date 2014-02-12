#!/bin/bash

#-------------------------------------------
#  
# Prototype script to checkout, compile Dakota
# 
#
# BvBW  10/06/08
# AGS  04/09
#-------------------------------------------

#-------------------------------------------
# setup and housekeeping
#-------------------------------------------

if [ -a $NIGHTLYDIR/dakota-stable.src.tar ]; then \rm $NIGHTLYDIR/dakota-stable.src.tar
fi
if [ -a $NIGHTLYDIR/dakota-stable.src.tar.gz ]; then \rm $NIGHTLYDIR/dakota-stable.src.tar.gz
fi

if [ -a $DAKOUTDIR ]; then \rm -rf $DAKOUTDIR
fi
mkdir  $DAKOUTDIR

cd $NIGHTLYDIR

#-------------------------------------------
# copy, configure, build, install Dakota 
#-------------------------------------------

wget -nv --no-check-certificate \
 https://development.sandia.gov/dakota/distributions/dakota/stable/dakota-stable.src.tar.gz \
 >  $DAKOUTDIR/dakota_wget.out 
gunzip dakota-stable.src.tar.gz

cd $TRIKOTADIR

tar xvf $NIGHTLYDIR/dakota-stable.src.tar > $DAKOUTDIR/dakota_untar.out 2>&1

mv dakota-5.4.0.src Dakota
