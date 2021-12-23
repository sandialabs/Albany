#!/bin/bash


awk '/cmake/{p=1;next}{if(p){print}}' do-cmake-trilinos-mpi-camobap-longdouble-eti >& /mnt/encrypted_sdc1/nightlyCDash/cdash-frag-ld-eti.txt
sed -i "s/\"/'/g" /mnt/encrypted_sdc1/nightlyCDash/cdash-frag-ld-eti.txt
sed -i 's/\.\.//g' /mnt/encrypted_sdc1/nightlyCDash/cdash-frag-ld-eti.txt
sed -i 's,\\,,g' /mnt/encrypted_sdc1/nightlyCDash/cdash-frag-ld-eti.txt
sed -i '/^$/d' /mnt/encrypted_sdc1/nightlyCDash/cdash-frag-ld-eti.txt
sed -i 's/-D /"-D/g' /mnt/encrypted_sdc1/nightlyCDash/cdash-frag-ld-eti.txt
awk '{print $0 "\""}' /mnt/encrypted_sdc1/nightlyCDash/cdash-frag-ld-eti.txt >& tmp.txt
mv tmp.txt /mnt/encrypted_sdc1/nightlyCDash/cdash-frag-ld-eti.txt
sed -i 's, \",\",g' /mnt/encrypted_sdc1/nightlyCDash/cdash-frag-ld-eti.txt
sed -i '$ d' /mnt/encrypted_sdc1/nightlyCDash/cdash-frag-ld-eti.txt
sed -i "s,{SEMS,ENV{SEMS,g" /mnt/encrypted_sdc1/nightlyCDash/cdash-frag-ld-eti.txt
cat /mnt/encrypted_sdc1/nightlyCDash/cdash-frag-ld-eti.txt
