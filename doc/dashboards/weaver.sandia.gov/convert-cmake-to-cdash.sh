#!/bin/bash


awk '/cmake/{p=1;next}{if(p){print}}' do-cmake-weaver-trilinos >& /home/projects/albany/nightlyCDashWeaver/cdash-frag.txt
sed -i "s/\"/'/g" /home/projects/albany/nightlyCDashWeaver/cdash-frag.txt
sed -i 's/\.\.//g' /home/projects/albany/nightlyCDashWeaver/cdash-frag.txt
sed -i 's,\\,,g' /home/projects/albany/nightlyCDashWeaver/cdash-frag.txt
sed -i '/^$/d' /home/projects/albany/nightlyCDashWeaver/cdash-frag.txt
sed -i 's/-D /"-D/g' /home/projects/albany/nightlyCDashWeaver/cdash-frag.txt
awk '{print $0 "\""}' /home/projects/albany/nightlyCDashWeaver/cdash-frag.txt >& tmp.txt
mv tmp.txt /home/projects/albany/nightlyCDashWeaver/cdash-frag.txt
sed -i 's, \",\",g' /home/projects/albany/nightlyCDashWeaver/cdash-frag.txt
sed -i '$ d' /home/projects/albany/nightlyCDashWeaver/cdash-frag.txt
sed -i "s,{SEMS,ENV{SEMS,g" /home/projects/albany/nightlyCDashWeaver/cdash-frag.txt
cat /home/projects/albany/nightlyCDashWeaver/cdash-frag.txt
