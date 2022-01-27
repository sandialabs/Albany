#!/bin/bash


awk '/cmake/{p=1;next}{if(p){print}}' do-cmake-trilinos-mpi-camobap-longdouble-eti >& /nightlyCDash/cdash-frag-ld-eti.txt
sed -i "s/\"/'/g" /nightlyCDash/cdash-frag-ld-eti.txt
sed -i 's/\.\.//g' /nightlyCDash/cdash-frag-ld-eti.txt
sed -i 's,\\,,g' /nightlyCDash/cdash-frag-ld-eti.txt
sed -i '/^$/d' /nightlyCDash/cdash-frag-ld-eti.txt
sed -i 's/-D /"-D/g' /nightlyCDash/cdash-frag-ld-eti.txt
awk '{print $0 "\""}' /nightlyCDash/cdash-frag-ld-eti.txt >& tmp.txt
mv tmp.txt /nightlyCDash/cdash-frag-ld-eti.txt
sed -i 's, \",\",g' /nightlyCDash/cdash-frag-ld-eti.txt
sed -i '$ d' /nightlyCDash/cdash-frag-ld-eti.txt
sed -i "s,{SEMS,ENV{SEMS,g" /nightlyCDash/cdash-frag-ld-eti.txt
cat /nightlyCDash/cdash-frag-ld-eti.txt
