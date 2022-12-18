#!/bin/bash

if [ "$1" == "" ]; then
  echo "Argument 1 (case num) not provided!" 
  exit 0
fi

build=$1

if [ $build = "regular" ]; then
  name="albany"
fi

awk '/cmake/{p=1;next}{if(p){print}}' do-cmake-$name >& /projects/albany/nightlyAlbanyCDash/cdash-$name-frag.txt
sed -i "s/\"/'/g" /projects/albany/nightlyAlbanyCDash/cdash-$name-frag.txt
sed -i 's/\.\.//g' /projects/albany/nightlyAlbanyCDash/cdash-$name-frag.txt
sed -i 's,\\,,g' /projects/albany/nightlyAlbanyCDash/cdash-$name-frag.txt
sed -i '/^$/d' /projects/albany/nightlyAlbanyCDash/cdash-$name-frag.txt
sed -i 's/-D /"-D/g' /projects/albany/nightlyAlbanyCDash/cdash-$name-frag.txt
awk '{print $0 "\""}' /projects/albany/nightlyAlbanyCDash/cdash-$name-frag.txt >& tmp.txt
mv tmp.txt /projects/albany/nightlyAlbanyCDash/cdash-$name-frag.txt
sed -i 's, \",\",g' /projects/albany/nightlyAlbanyCDash/cdash-$name-frag.txt
sed -i '$ d' /projects/albany/nightlyAlbanyCDash/cdash-$name-frag.txt
sed -i 's/-G/\"-G/g' /projects/albany/nightlyAlbanyCDash/cdash-$name-frag.txt
sed -i 's/-W/\"-W/g' /projects/albany/nightlyAlbanyCDash/cdash-$name-frag.txt
cat /projects/albany/nightlyAlbanyCDash/cdash-$name-frag.txt
