#!/bin/bash

awk '/cmake/{p=1;next}{if(p){print}}' do-cmake-cori-trilinos >& /project/projectdirs/piscees/nightlyCoriCDash/cdash-frag.txt
sed -i "s/\"/'/g" /project/projectdirs/piscees/nightlyCoriCDash/cdash-frag.txt
sed -i 's/\.\.//g' /project/projectdirs/piscees/nightlyCoriCDash/cdash-frag.txt
sed -i 's,\\,,g' /project/projectdirs/piscees/nightlyCoriCDash/cdash-frag.txt
sed -i '/^$/d' /project/projectdirs/piscees/nightlyCoriCDash/cdash-frag.txt
sed -i 's/-D /"-D/g' /project/projectdirs/piscees/nightlyCoriCDash/cdash-frag.txt
awk '{print $0 "\""}' /project/projectdirs/piscees/nightlyCoriCDash/cdash-frag.txt >& tmp.txt
mv tmp.txt /project/projectdirs/piscees/nightlyCoriCDash/cdash-frag.txt
sed -i 's, \",\",g' /project/projectdirs/piscees/nightlyCoriCDash/cdash-frag.txt
sed -i '$ d' /project/projectdirs/piscees/nightlyCoriCDash/cdash-frag.txt
sed -i "s,{SEMS,ENV{SEMS,g" /project/projectdirs/piscees/nightlyCoriCDash/cdash-frag.txt
cat /project/projectdirs/piscees/nightlyCoriCDash/cdash-frag.txt
