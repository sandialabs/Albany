#!/bin/bash

#if [ "$1" == "" ]; then
#  echo "Argument 1 (case num) not provided!" 
#  exit 0
#fi
#if [ "$2" == "" ]; then
#  echo "Argument 2 (case num) not provided!" 
#  exit 0
#fi

#compiler=$1
#kokkosnode=$2

awk '/cmake/{p=1;next}{if(p){print}}' do-cmake-trilinos-mpi-hayka >& /home/ikalash/nightlyAlbanyCDash/cdash-hayka-frag.txt
sed -i "s/\"/'/g" /home/ikalash/nightlyAlbanyCDash/cdash-hayka-frag.txt
sed -i 's/\.\.//g' /home/ikalash/nightlyAlbanyCDash/cdash-hayka-frag.txt
sed -i 's,\\,,g' /home/ikalash/nightlyAlbanyCDash/cdash-hayka-frag.txt
sed -i '/^$/d' /home/ikalash/nightlyAlbanyCDash/cdash-hayka-frag.txt
sed -i 's/-D /"-D/g' /home/ikalash/nightlyAlbanyCDash/cdash-hayka-frag.txt
awk '{print $0 "\""}' /home/ikalash/nightlyAlbanyCDash/cdash-hayka-frag.txt >& tmp.txt
mv tmp.txt /home/ikalash/nightlyAlbanyCDash/cdash-hayka-frag.txt
sed -i 's, \",\",g' /home/ikalash/nightlyAlbanyCDash/cdash-hayka-frag.txt
sed -i '$ d' /home/ikalash/nightlyAlbanyCDash/cdash-hayka-frag.txt
sed -i "s,{OPEN,ENV{OPEN,g" /home/ikalash/nightlyAlbanyCDash/cdash-hayka-frag.txt
sed -i "s,{NETCDF,ENV{NETCDF,g" /home/ikalash/nightlyAlbanyCDash/cdash-hayka-frag.txt
sed -i "s,{PAR,ENV{PAR,g" /home/ikalash/nightlyAlbanyCDash/cdash-hayka-frag.txt
sed -i "s,{HDF,ENV{HDF,g" /home/ikalash/nightlyAlbanyCDash/cdash-hayka-frag.txt
#sed -i "s,{,ENV{,g" /home/ikalash/nightlyAlbanyCDash/cdash-hayka-frag.txt
#sed -i "s,{INSTALL,ENV{INSTALL,g" /home/ikalash/nightlyAlbanyCDash/cdash-hayka-frag.txt
cat /home/ikalash/nightlyAlbanyCDash/cdash-hayka-frag.txt
