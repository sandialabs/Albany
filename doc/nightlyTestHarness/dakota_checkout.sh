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

#New for Dakota 6.2
wget -nv --no-check-certificate https://dakota.sandia.gov/sites/default/files/distributions/public/dakota-6.2-public.src.tar.gz >  $DAKOUTDIR/dakota_wget.out 
cd $TRIKOTADIR
tar xvfz $NIGHTLYDIR/dakota-6.2-public.src.tar.gz  > $DAKOUTDIR/dakota_untar.out 2>&1
mv dakota-6.2.0.src Dakota

