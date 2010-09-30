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

if [ -a $NIGHTLYDIR/Dakota_votd.src.tar ]; then \rm $NIGHTLYDIR/Dakota_votd.src.tar
fi
if [ -a $NIGHTLYDIR/Dakota_votd.src.tar.gz ]; then \rm $NIGHTLYDIR/Dakota_votd.src.tar.gz
fi

if [ -a $DAKOUTDIR ]; then \rm -rf $DAKOUTDIR
fi
mkdir  $DAKOUTDIR

cd $NIGHTLYDIR

#-------------------------------------------
# copy, configure, build, install Dakota 
#-------------------------------------------

wget -nv --no-check-certificate \
 https://development.sandia.gov/DAKOTA/licensing/votd/Dakota_votd.src.tar.gz \
 >  $DAKOUTDIR/dakota_wget.out 
gunzip Dakota_votd.src.tar.gz

cd $TRIKOTADIR

tar xvf $NIGHTLYDIR/Dakota_votd.src.tar > $DAKOUTDIR/dakota_untar.out 2>&1
