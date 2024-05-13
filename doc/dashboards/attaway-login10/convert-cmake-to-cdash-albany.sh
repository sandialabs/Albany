#!/bin/bash

if [ "$1" == "" ]; then
  echo "Argument 1 (case num) not provided!" 
  exit 0
fi

build=$1

if [ $build = "regular" ]; then
  name="albany"
fi
if [ $build = "no-epetra" ]; then
  name="albanyT"
fi
if [ $build = "fpe" ]; then
  name="albanyFPE"
fi
if [ $build = "openmp" ]; then
  name="albanyOpenmp"
fi
if [ $build = "cali" ]; then
  name="cismAlbany"
fi

awk '/cmake/{p=1;next}{if(p){print}}' do-cmake-$name-attaway-intel-serial >& /projects/albany/nightlyCDash/cdash-$name-frag.txt
sed -i "s/\"/'/g" /projects/albany/nightlyCDash/cdash-$name-frag.txt
sed -i 's/\.\.//g' /projects/albany/nightlyCDash/cdash-$name-frag.txt
sed -i 's,\\,,g' /projects/albany/nightlyCDash/cdash-$name-frag.txt
sed -i '/^$/d' /projects/albany/nightlyCDash/cdash-$name-frag.txt
sed -i 's/-D /"-D/g' /projects/albany/nightlyCDash/cdash-$name-frag.txt
awk '{print $0 "\""}' /projects/albany/nightlyCDash/cdash-$name-frag.txt >& tmp.txt
mv tmp.txt /projects/albany/nightlyCDash/cdash-$name-frag.txt
sed -i 's, \",\",g' /projects/albany/nightlyCDash/cdash-$name-frag.txt
#sed -i '$ d' /projects/albany/nightlyCDash/cdash-$name-frag.txt
sed -i 's/-G/\"-G/g' /projects/albany/nightlyCDash/cdash-$name-frag.txt
sed -i 's/-W/\"-W/g' /projects/albany/nightlyCDash/cdash-$name-frag.txt
cat /projects/albany/nightlyCDash/cdash-$name-frag.txt
