#!/bin/bash

BASE_DIR=/pscratch/sd/m/mcarlson/biweeklyCDashPerlmutter-cuda

awk '/cmake/{p=1;next}{if(p){print}}' do-cmake-pm_gpu-trilinos >& ${BASE_DIR}/cdash-frag.txt
sed -i "s/\"/'/g" ${BASE_DIR}/cdash-frag.txt
sed -i 's,\\,,g' ${BASE_DIR}/cdash-frag.txt
sed -i '/^$/d' ${BASE_DIR}/cdash-frag.txt
sed -i 's/-D /"-D/g' ${BASE_DIR}/cdash-frag.txt
awk '{print $0 "\""}' ${BASE_DIR}/cdash-frag.txt >& tmp.txt
mv tmp.txt ${BASE_DIR}/cdash-frag.txt
sed -i 's, \",\",g' ${BASE_DIR}/cdash-frag.txt
sed -i '$ d' ${BASE_DIR}/cdash-frag.txt
sed -i "s,{SEMS,ENV{SEMS,g" ${BASE_DIR}/cdash-frag.txt
cat ${BASE_DIR}/cdash-frag.txt
