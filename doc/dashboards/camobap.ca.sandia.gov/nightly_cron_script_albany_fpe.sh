#!/bin/sh

cd /mnt/encrypted_sdc1/nightlyCDash

#unset http_proxy
#unset https_proxy

export https_proxy="http://proxy.ca.sandia.gov:80"
export http_proxy="http://proxy.ca.sandia.gov:80"

LOG_FILE=/mnt/encrypted_sdc1/nightlyCDash/nightly_logFPE.txt

eval "env  TEST_DIRECTORY=/mnt/encrypted_sdc1/nightlyCDash SCRIPT_DIRECTORY=/mnt/encrypted_sdc1/nightlyCDash ctest -VV -S /mnt/encrypted_sdc1/nightlyCDash/ctest_nightly_albanyFPE.cmake" > $LOG_FILE 2>&1

# Copy a basic installation to /projects/albany for those who like a nightly
# build.
#cp -r build/TrilinosInstall/* /projects/albany/trilinos/nightly/;
#chmod -R a+X /projects/albany/trilinos/nightly;
#chmod -R a+r /projects/albany/trilinos/nightly;
