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
kokkosnode=$2

awk '/cmake/{p=1;next}{if(p){print}}' do-cmake-trilinos-attaway-$compiler-$kokkosnode >& /projects/albany/nightlyCDash/cdash-$compiler-$kokkosnode-frag.txt
sed -i "s/\"/'/g" /projects/albany/nightlyCDash/cdash-$compiler-$kokkosnode-frag.txt
sed -i 's/\.\.//g' /projects/albany/nightlyCDash/cdash-$compiler-$kokkosnode-frag.txt
sed -i 's,\\,,g' /projects/albany/nightlyCDash/cdash-$compiler-$kokkosnode-frag.txt
sed -i '/^$/d' /projects/albany/nightlyCDash/cdash-$compiler-$kokkosnode-frag.txt
sed -i 's/-D /"-D/g' /projects/albany/nightlyCDash/cdash-$compiler-$kokkosnode-frag.txt
awk '{print $0 "\""}' /projects/albany/nightlyCDash/cdash-$compiler-$kokkosnode-frag.txt >& tmp.txt
mv tmp.txt /projects/albany/nightlyCDash/cdash-$compiler-$kokkosnode-frag.txt
sed -i 's, \",\",g' /projects/albany/nightlyCDash/cdash-$compiler-$kokkosnode-frag.txt
#sed -i '$ d' /projects/albany/nightlyCDash/cdash-$compiler-$kokkosnode-frag.txt
#sed -i "s,{SEMS,ENV{SEMS,g" /projects/albany/nightlyCDash/cdash-$compiler-$kokkosnode-frag.txt
sed -i "s,{,ENV{,g" /projects/albany/nightlyCDash/cdash-$compiler-$kokkosnode-frag.txt
sed -i "s,ENV{INSTALL,{INSTALL,g" /projects/albany/nightlyCDash/cdash-$compiler-$kokkosnode-frag.txt
sed -i "s,ENV{BTYPE,{BTYPE,g" /projects/albany/nightlyCDash/cdash-$compiler-$kokkosnode-frag.txt
cat /projects/albany/nightlyCDash/cdash-$compiler-$kokkosnode-frag.txt
