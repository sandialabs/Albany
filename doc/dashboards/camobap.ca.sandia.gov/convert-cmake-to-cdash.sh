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

awk '/cmake/{p=1;next}{if(p){print}}' do-cmake-$name >& /nightlyCDash/cdash-$name-frag.txt
sed -i "s/\"/'/g" /nightlyCDash/cdash-$name-frag.txt
sed -i 's/\.\.//g' /nightlyCDash/cdash-$name-frag.txt
sed -i 's,\\,,g' /nightlyCDash/cdash-$name-frag.txt
sed -i '/^$/d' /nightlyCDash/cdash-$name-frag.txt
sed -i 's/-D /"-D/g' /nightlyCDash/cdash-$name-frag.txt
awk '{print $0 "\""}' /nightlyCDash/cdash-$name-frag.txt >& tmp.txt
mv tmp.txt /nightlyCDash/cdash-$name-frag.txt
sed -i 's, \",\",g' /nightlyCDash/cdash-$name-frag.txt
sed -i 's/-G/\"-G/g' /nightlyCDash/cdash-$name-frag.txt
sed -i 's/-W/\"-W/g' /nightlyCDash/cdash-$name-frag.txt
#sed -i '$ d' /nightlyCDash/cdash-$name-frag.txt
sed -i 's,/",,g' /nightlyCDash/cdash-$name-frag.txt
cat /nightlyCDash/cdash-$name-frag.txt
