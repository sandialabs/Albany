#!/bin/bash

rm -rf TriBITS
rm -rf test_history
rm -rf *html
rm -rf *out
rm -rf *json
rm -rf  albanyNightlyBuildsTwoif.csv 

git clone git@github.com:TriBITSPub/TriBITS.git


now=$(date +"%Y-%m-%d")

#./albany_cdash_status_my.cdash.org.sh --date=$now --email-from-address=ikalash@cee-compute003.sandia.gov --send-email-to=ikalash@sandia.gov,mperego@sandia.gov,jwatkin@sandia.gov,lbertag@sandia.gov,maxcarl@sandia.gov 
./albany_cdash_status_my.cdash.org.sh --date=$now --email-from-address=ikalash@cee-compute003.sandia.gov --send-email-to=ikalash@sandia.gov


