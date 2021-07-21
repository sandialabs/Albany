#!/bin/bash


awk '/cmake/{p=1;next}{if(p){print}}' do-cmake-trilinos-mpi-camobap-extended-sts >& /nightlyCDash/cdash-frag.txt
sed -i "s/\"/'/g" /nightlyCDash/cdash-frag.txt
sed -i 's/\.\.//g' /nightlyCDash/cdash-frag.txt
sed -i 's,\\,,g' /nightlyCDash/cdash-frag.txt
sed -i '/^$/d' /nightlyCDash/cdash-frag.txt
sed -i 's/-D /"-D/g' /nightlyCDash/cdash-frag.txt
awk '{print $0 "\""}' /nightlyCDash/cdash-frag.txt >& tmp.txt
mv tmp.txt /nightlyCDash/cdash-frag.txt
sed -i 's, \",\",g' /nightlyCDash/cdash-frag.txt
sed -i '$ d' /nightlyCDash/cdash-frag.txt
sed -i "s,{SEMS,ENV{SEMS,g" /nightlyCDash/cdash-frag.txt
cat /nightlyCDash/cdash-frag.txt
