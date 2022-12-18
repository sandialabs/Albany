#!/bin/bash

if [ "$1" == "" ]; then
  echo "Argument 1 (case num) not provided!" 
  exit 0
fi

build=$1

if [ $build = "regular" ]; then
  name="albany"
fi
if [ $build = "cali" ]; then
  name="cism-albany"
fi

awk '/cmake/{p=1;next}{if(p){print}}' do-cmake-$name >& /project/projectdirs/piscees/nightlyCoriCDash/cdash-$name-frag.txt
sed -i "s/\"/'/g" /project/projectdirs/piscees/nightlyCoriCDash/cdash-$name-frag.txt
sed -i 's/\.\.//g' /project/projectdirs/piscees/nightlyCoriCDash/cdash-$name-frag.txt
sed -i 's,\\,,g' /project/projectdirs/piscees/nightlyCoriCDash/cdash-$name-frag.txt
sed -i '/^$/d' /project/projectdirs/piscees/nightlyCoriCDash/cdash-$name-frag.txt
awk '{print $0 "\""}' /project/projectdirs/piscees/nightlyCoriCDash/cdash-$name-frag.txt >& tmp.txt
mv tmp.txt /project/projectdirs/piscees/nightlyCoriCDash/cdash-$name-frag.txt
sed -i 's, \",\",g' /project/projectdirs/piscees/nightlyCoriCDash/cdash-$name-frag.txt
sed -i '$ d' /project/projectdirs/piscees/nightlyCoriCDash/cdash-$name-frag.txt
sed -i 's/-G/\"-G/g' /project/projectdirs/piscees/nightlyCoriCDash/cdash-$name-frag.txt
sed -i 's/-Wno-dev/\"-Wno-dev/g' /project/projectdirs/piscees/nightlyCoriCDash/cdash-$name-frag.txt
sed -i "s,='\",=',g" /project/projectdirs/piscees/nightlyCoriCDash/cdash-$name-frag.txt
sed -i 's/-D /"-D/g' /project/projectdirs/piscees/nightlyCoriCDash/cdash-$name-frag.txt
cat /project/projectdirs/piscees/nightlyCoriCDash/cdash-$name-frag.txt
