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

awk '/cmake/{p=1;next}{if(p){print}}' do-cmake-$name >& /home/ikalash/nightlyAlbanyCDash/cdash-$name-frag.txt
sed -i "s/\"/'/g" /home/ikalash/nightlyAlbanyCDash/cdash-$name-frag.txt
sed -i 's/\.\.//g' /home/ikalash/nightlyAlbanyCDash/cdash-$name-frag.txt
sed -i 's,\\,,g' /home/ikalash/nightlyAlbanyCDash/cdash-$name-frag.txt
sed -i '/^$/d' /home/ikalash/nightlyAlbanyCDash/cdash-$name-frag.txt
sed -i 's/-D /"-D/g' /home/ikalash/nightlyAlbanyCDash/cdash-$name-frag.txt
awk '{print $0 "\""}' /home/ikalash/nightlyAlbanyCDash/cdash-$name-frag.txt >& tmp.txt
mv tmp.txt /home/ikalash/nightlyAlbanyCDash/cdash-$name-frag.txt
sed -i 's, \",\",g' /home/ikalash/nightlyAlbanyCDash/cdash-$name-frag.txt
sed -i '$ d' /home/ikalash/nightlyAlbanyCDash/cdash-$name-frag.txt
sed -i 's/-G/\"-G/g' /home/ikalash/nightlyAlbanyCDash/cdash-$name-frag.txt
sed -i 's/-W/\"-W/g' /home/ikalash/nightlyAlbanyCDash/cdash-$name-frag.txt
sed -i "s,{,ENV{,g" /home/ikalash/nightlyAlbanyCDash/cdash-$name-frag.txt
sed -i "s,ENV{ALB,{ALB,g" /home/ikalash/nightlyAlbanyCDash/cdash-$name-frag.txt
sed -i "s,ENV{TRI,{TRI,g" /home/ikalash/nightlyAlbanyCDash/cdash-$name-frag.txt
sed -i "s,ENV{CISM,{CISM,g" /home/ikalash/nightlyAlbanyCDash/cdash-$name-frag.txt
cat /home/ikalash/nightlyAlbanyCDash/cdash-$name-frag.txt
