#!/bin/bash

if [ "$1" == "" ]; then
  echo "Argument 1 (case num) not provided!" 
  exit 0
fi

build=$1

if [ $build = "intel-serial" ]; then
  name="albany-intel-serial"
fi
if [ $build = "intel-openmp" ]; then
  name="albany-intel-openmp"
fi
if [ $build = "gcc-serial" ]; then
  name="albany-gcc-serial"
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
sed -i 's/-W/\"-W/g' ${DIR}/cdash-$name-frag.txt
cat ${DIR}/cdash-$name-frag.txt
