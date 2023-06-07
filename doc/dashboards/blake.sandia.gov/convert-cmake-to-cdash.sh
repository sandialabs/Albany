#!/bin/bash

if [ "$1" == "" ]; then
  echo "Argument 1 (case num) not provided!" 
  exit 0
fi
if [ "$2" == "" ]; then
  echo "Argument 2 (case num) not provided!" 
  exit 0
fi

compiler=$1
buildtype=$2

awk '/cmake/{p=1;next}{if(p){print}}' do-cmake-trilinos-$compiler-$buildtype >& /home/projects/albany/nightlyCDashTrilinosBlake/cdash-$compiler-$buildtype-frag.txt
sed -i "s/\"/'/g" /home/projects/albany/nightlyCDashTrilinosBlake/cdash-$compiler-$buildtype-frag.txt
sed -i 's/\.\.//g' /home/projects/albany/nightlyCDashTrilinosBlake/cdash-$compiler-$buildtype-frag.txt
sed -i 's,\\,,g' /home/projects/albany/nightlyCDashTrilinosBlake/cdash-$compiler-$buildtype-frag.txt
sed -i '/^$/d' /home/projects/albany/nightlyCDashTrilinosBlake/cdash-$compiler-$buildtype-frag.txt
sed -i 's/-D /"-D/g' /home/projects/albany/nightlyCDashTrilinosBlake/cdash-$compiler-$buildtype-frag.txt
awk '{print $0 "\""}' /home/projects/albany/nightlyCDashTrilinosBlake/cdash-$compiler-$buildtype-frag.txt >& tmp.txt
mv tmp.txt /home/projects/albany/nightlyCDashTrilinosBlake/cdash-$compiler-$buildtype-frag.txt
sed -i 's, \",\",g' /home/projects/albany/nightlyCDashTrilinosBlake/cdash-$compiler-$buildtype-frag.txt
sed -i '$ d' /home/projects/albany/nightlyCDashTrilinosBlake/cdash-$compiler-$buildtype-frag.txt
sed -i "s,{SEMS,ENV{SEMS,g" /home/projects/albany/nightlyCDashTrilinosBlake/cdash-$compiler-$buildtype-frag.txt
cat /home/projects/albany/nightlyCDashTrilinosBlake/cdash-$compiler-$buildtype-frag.txt
