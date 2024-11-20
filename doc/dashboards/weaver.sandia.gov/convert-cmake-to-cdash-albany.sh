#!/bin/bash

if [ "$1" == "" ]; then
  echo "Argument 1 (case num) not provided!" 
  exit 0
fi

build=$1

if [ $build = "regular" ]; then
  name="albany"
fi
if [ $build = "sfad" ]; then
  name="albanySFAD"
fi

awk '/cmake/{p=1;next}{if(p){print}}' do-cmake-$name >& /home/projects/albany/nightlyCDashWeaver/cdash-$name-frag.txt
sed -i "s/\"/'/g" /home/projects/albany/nightlyCDashWeaver/cdash-$name-frag.txt
sed -i 's/\.\.//g' /home/projects/albany/nightlyCDashWeaver/cdash-$name-frag.txt
sed -i 's,\\,,g' /home/projects/albany/nightlyCDashWeaver/cdash-$name-frag.txt
sed -i '/^$/d' /home/projects/albany/nightlyCDashWeaver/cdash-$name-frag.txt
sed -i 's/-D /"-D/g' /home/projects/albany/nightlyCDashWeaver/cdash-$name-frag.txt
awk '{print $0 "\""}' /home/projects/albany/nightlyCDashWeaver/cdash-$name-frag.txt >& tmp.txt
mv tmp.txt /home/projects/albany/nightlyCDashWeaver/cdash-$name-frag.txt
sed -i 's, \",\",g' /home/projects/albany/nightlyCDashWeaver/cdash-$name-frag.txt
#sed -i '$ d' /home/projects/albany/nightlyCDashWeaver/cdash-$name-frag.txt
sed -i 's/-G/\"-G/g' /home/projects/albany/nightlyCDashWeaver/cdash-$name-frag.txt
sed -i 's/-W/\"-W/g' /home/projects/albany/nightlyCDashWeaver/cdash-$name-frag.txt
cat /home/projects/albany/nightlyCDashWeaver/cdash-$name-frag.txt
