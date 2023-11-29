#!/bin/bash

if [ "$1" == "" ]; then
  echo "Argument 1 (case num) not provided!" 
  exit 0
fi

build=$1

if [ $build = "gcc-release" ]; then
  name="albany-gcc-release"
fi
if [ $build = "gcc-debug" ]; then
  name="albany-gcc-debug"
fi
if [ $build = "intel-release" ]; then
  name="albany-intel-release"
fi
if [ $build = "sfad" ]; then
  name="albany-sfad"
fi

DIR=/home/projects/albany/nightlyCDashAlbanyBlake
#DIR=`pwd`
awk '/cmake/{p=1;next}{if(p){print}}' do-cmake-$name >& ${DIR}/cdash-$name-frag.txt
sed -i "s/\"/'/g" ${DIR}/cdash-$name-frag.txt
sed -i 's/\.\.//g' ${DIR}/cdash-$name-frag.txt
sed -i 's,\\,,g' ${DIR}/cdash-$name-frag.txt
sed -i '/^$/d' ${DIR}/cdash-$name-frag.txt
sed -i 's/-D /"-D/g' ${DIR}/cdash-$name-frag.txt
awk '{print $0 "\""}' ${DIR}/cdash-$name-frag.txt >& tmp.txt
mv tmp.txt ${DIR}/cdash-$name-frag.txt
sed -i 's, \",\",g' ${DIR}/cdash-$name-frag.txt
sed -i 's/-G/\"-G/g' ${DIR}/cdash-$name-frag.txt
#sed -i 's/-W/\"-W/g' ${DIR}/cdash-$name-frag.txt
cat ${DIR}/cdash-$name-frag.txt
