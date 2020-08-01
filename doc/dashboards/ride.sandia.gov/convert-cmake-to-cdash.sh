#!/bin/bash

awk '/cmake/{p=1;next}{if(p){print}}' do-cmake-trilinos >& cdash-frag.txt
sed -i "s/\"/'/g" cdash-frag.txt
sed -i 's/\.\.//g' cdash-frag.txt
sed -i 's,\\,,g' cdash-frag.txt
sed -i '/^$/d' cdash-frag.txt
sed -i 's/-D /"-D/g' cdash-frag.txt
awk '{print $0 "\""}' cdash-frag.txt >& tmp.txt
mv tmp.txt cdash-frag.txt
sed -i 's, \",\",g' cdash-frag.txt
sed -i '$ d' cdash-frag.txt
sed -i "s,{SEMS,ENV{SEMS,g" cdash-frag.txt
cat cdash-frag.txt
