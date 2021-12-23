#!/bin/bash


awk '/cmake/{p=1;next}{if(p){print}}' do-cmake-trilinos-mpi-camobap-extended-sts >& /mnt/encrypted_sdc1/nightlyCDash/cdash-frag.txt
sed -i "s/\"/'/g" /mnt/encrypted_sdc1/nightlyCDash/cdash-frag.txt
sed -i 's/\.\.//g' /mnt/encrypted_sdc1/nightlyCDash/cdash-frag.txt
sed -i 's,\\,,g' /mnt/encrypted_sdc1/nightlyCDash/cdash-frag.txt
sed -i '/^$/d' /mnt/encrypted_sdc1/nightlyCDash/cdash-frag.txt
sed -i 's/-D /"-D/g' /mnt/encrypted_sdc1/nightlyCDash/cdash-frag.txt
awk '{print $0 "\""}' /mnt/encrypted_sdc1/nightlyCDash/cdash-frag.txt >& tmp.txt
mv tmp.txt /mnt/encrypted_sdc1/nightlyCDash/cdash-frag.txt
sed -i 's, \",\",g' /mnt/encrypted_sdc1/nightlyCDash/cdash-frag.txt
sed -i '$ d' /mnt/encrypted_sdc1/nightlyCDash/cdash-frag.txt
sed -i "s,{SEMS,ENV{SEMS,g" /mnt/encrypted_sdc1/nightlyCDash/cdash-frag.txt
cat /mnt/encrypted_sdc1/nightlyCDash/cdash-frag.txt
