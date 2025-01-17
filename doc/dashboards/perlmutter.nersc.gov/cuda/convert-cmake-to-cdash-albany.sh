#!/bin/bash

FAD_CONFIGURATION=${1}
FAD_SIZE=${2}

BASE_DIR=/pscratch/sd/m/mcarlson/biweeklyCDashPerlmutter-cuda

if [ "$FAD_CONFIGURATION" = "slfad" ] ; then
  FRAG_NAME="${BASE_DIR}/cdash-albany-frag-slfad.txt"
  TMP_NAME="tmp_slfad.txt"
  awk '/cmake/{p=1;next}{if(p){print}}' do-cmake-pm_cpu-albany-slfad >& ${FRAG_NAME}
fi
if [ "$FAD_CONFIGURATION" = "sfad" ] && [ "$FAD_SIZE" = "12" ]; then
  FRAG_NAME="${BASE_DIR}/cdash-albany-frag-sfad12.txt"
  TMP_NAME="tmp_sfad12.txt"
  awk '/cmake/{p=1;next}{if(p){print}}' do-cmake-pm_cpu-albany-sfad >& ${FRAG_NAME}
fi
if [ "$FAD_CONFIGURATION" = "sfad" ] && [ "$FAD_SIZE" = "24" ]; then
  FRAG_NAME="${BASE_DIR}/cdash-albany-frag-sfad24.txt"
  TMP_NAME="tmp_sfad24.txt"
  awk '/cmake/{p=1;next}{if(p){print}}' do-cmake-pm_cpu-albany-sfad >& ${FRAG_NAME}
fi

sed -i "s/\"/'/g" ${FRAG_NAME}
sed -i 's,\\,,g' ${FRAG_NAME}
sed -i '/^$/d' ${FRAG_NAME}
sed -i 's/-D /"-D/g' ${FRAG_NAME}
awk '{print $0 "\""}' ${FRAG_NAME} >& ${TMP_NAME}
mv ${TMP_NAME} ${FRAG_NAME}
sed -i 's, \",\",g' ${FRAG_NAME}
sed -i 's/-G/\"-G/g' ${FRAG_NAME}
sed -i 's/-W/\"-W/g' ${FRAG_NAME}
cat ${FRAG_NAME}
