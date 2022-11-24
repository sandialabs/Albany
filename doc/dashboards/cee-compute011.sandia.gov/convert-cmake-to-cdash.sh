#!/bin/bash

if [ "$1" == "" ]; then
  echo "Argument 1 (case num) not provided!" 
  exit 0
fi

compiler=$1

awk '/cmake/{p=1;next}{if(p){print}}' do-cmake-trilinos-mpi-sems-$compiler >& /projects/albany/nightlyAlbanyCDash/cdash-$compiler-frag.txt
sed -i "s/\"/'/g" /projects/albany/nightlyAlbanyCDash/cdash-$compiler-frag.txt
sed -i 's/\.\.//g' /projects/albany/nightlyAlbanyCDash/cdash-$compiler-frag.txt
sed -i 's,\\,,g' /projects/albany/nightlyAlbanyCDash/cdash-$compiler-frag.txt
sed -i '/^$/d' /projects/albany/nightlyAlbanyCDash/cdash-$compiler-frag.txt
sed -i 's/-D /"-D/g' /projects/albany/nightlyAlbanyCDash/cdash-$compiler-frag.txt
awk '{print $0 "\""}' /projects/albany/nightlyAlbanyCDash/cdash-$compiler-frag.txt >& tmp.txt
mv tmp.txt /projects/albany/nightlyAlbanyCDash/cdash-$compiler-frag.txt
sed -i 's, \",\",g' /projects/albany/nightlyAlbanyCDash/cdash-$compiler-frag.txt
sed -i '$ d' /projects/albany/nightlyAlbanyCDash/cdash-$compiler-frag.txt
#sed -i "s,{SEMS,ENV{SEMS,g" /projects/albany/nightlyAlbanyCDash/cdash-$compiler-frag.txt
#sed -i "s,{MPI,ENV{MPI,g" /projects/albany/nightlyAlbanyCDash/cdash-$compiler-frag.txt
sed -i "s,{,ENV{,g" /projects/albany/nightlyAlbanyCDash/cdash-$compiler-frag.txt
sed -i "s,ENV{INSTALL,{INSTALL,g" /projects/albany/nightlyAlbanyCDash/cdash-$compiler-frag.txt
sed -i "s,ENV{BTYPE,{BTYPE,g" /projects/albany/nightlyAlbanyCDash/cdash-$compiler-frag.txt
cat /projects/albany/nightlyAlbanyCDash/cdash-$compiler-frag.txt
